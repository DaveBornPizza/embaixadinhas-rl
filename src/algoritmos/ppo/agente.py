"""
PPO (Proximal Policy Optimization) escrito à mão com PyTorch, para ações contínuas.

Peças:
    RedeAtorCritico   política (ator) + função valor (crítico) em duas redes separadas
    NormalizadorRecompensa   escala as recompensas pelo desvio-padrão do retorno
    AgentePPO         coleta o rollout, calcula as vantagens (GAE) e atualiza a rede

Como funciona, em resumo:
    1. A política é uma Gaussiana: a rede dá a média de cada torque e há um
       log(desvio-padrão) aprendido, que não depende do estado.
    2. O agente joga `passos_rollout` passos guardando (s, a, log π(a|s), V(s), r).
    3. As vantagens saem do GAE (Generalized Advantage Estimation).
    4. A política é atualizada por várias épocas de minibatches, com a razão
       r(θ) = π_novo(a|s) / π_antigo(a|s) cortada em [1 - ε, 1 + ε]. O corte impede
       que um único passo de gradiente mude demais a política.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


def camada(entrada, saida, ganho=np.sqrt(2)):
    """Linear com inicialização ortogonal (padrão em PPO) e viés zero."""
    c = nn.Linear(entrada, saida)
    nn.init.orthogonal_(c.weight, ganho)
    nn.init.zeros_(c.bias)
    return c


class RedeAtorCritico(nn.Module):
    """Entrada: estado (10 números).
    Ator   -> média de cada torque (3 números).
    Crítico -> V(s), um único número.
    """

    def __init__(self, dim_obs, dim_acao, oculta=64, log_std_inicial=-0.5):
        super().__init__()
        self.ator = nn.Sequential(
            camada(dim_obs, oculta), nn.Tanh(),
            camada(oculta, oculta), nn.Tanh(),
            camada(oculta, dim_acao, ganho=0.01),     # ganho pequeno: começa quase em média 0
        )
        self.critico = nn.Sequential(
            camada(dim_obs, oculta), nn.Tanh(),
            camada(oculta, oculta), nn.Tanh(),
            camada(oculta, 1, ganho=1.0),
        )
        # log do desvio-padrão da Gaussiana: um parâmetro por torque, independente do estado
        self.log_std = nn.Parameter(torch.full((dim_acao,), float(log_std_inicial)))

    def distribuicao(self, obs):
        media = self.ator(obs)
        std = self.log_std.clamp(-5.0, 2.0).exp().expand_as(media)
        return Normal(media, std)

    def valor(self, obs):
        return self.critico(obs).squeeze(-1)


class NormalizadorRecompensa:
    """Divide a recompensa pelo desvio-padrão do retorno descontado acumulado.

    As recompensas do ambiente vão de -10 a +10 e o retorno de um episódio passa de
    centenas; isso deixa a perda do crítico enorme. Escalar ajuda a estabilizar.
    (Só muda a escala, não o sinal nem a política ótima.)
    """

    def __init__(self, gamma, eps=1e-8):
        self.gamma = gamma
        self.eps = eps
        self.media, self.var, self.n = 0.0, 1.0, 1e-4
        self.retorno = 0.0

    def __call__(self, recompensa, fim_episodio):
        self.retorno = self.retorno * self.gamma + recompensa
        # atualização incremental de média e variância
        delta = self.retorno - self.media
        total = self.n + 1
        self.media += delta / total
        self.var = (self.var * self.n + delta ** 2 * self.n / total) / total
        self.n = total
        if fim_episodio:
            self.retorno = 0.0
        return float(np.clip(recompensa / np.sqrt(self.var + self.eps), -10.0, 10.0))


class AgentePPO:
    def __init__(self, dim_obs, dim_acao,
                 gamma=0.99,               # desconto
                 lam=0.95,                 # λ do GAE (viés x variância da vantagem)
                 lr=3e-4,
                 clip=0.2,                 # ε do corte da razão de probabilidades
                 epocas=10,                # quantas vezes o mesmo rollout é reaproveitado
                 tamanho_lote=64,          # minibatch
                 passos_rollout=2048,      # passos coletados antes de cada atualização
                 coef_valor=0.5,           # peso da perda do crítico
                 coef_entropia=0.0,        # bônus de entropia (exploração)
                 max_grad_norm=0.5,
                 oculta=64,
                 log_std_inicial=-1.0,     # desvio-padrão inicial = e^-1 ≈ 0,37 (antes: 0,61)
                 normalizar_recompensa=True):
        self.dim_obs, self.dim_acao = dim_obs, dim_acao
        self.gamma, self.lam = gamma, lam
        self.lr_inicial = lr
        self.clip = clip
        self.epocas = epocas
        self.tamanho_lote = tamanho_lote
        self.passos_rollout = passos_rollout
        self.coef_valor, self.coef_entropia = coef_valor, coef_entropia
        self.max_grad_norm = max_grad_norm

        self.rede = RedeAtorCritico(dim_obs, dim_acao, oculta, log_std_inicial)
        self.otimizador = torch.optim.Adam(self.rede.parameters(), lr=lr, eps=1e-5)
        self.norm_rec = NormalizadorRecompensa(gamma) if normalizar_recompensa else None
        self._limpar_rollout()

    # ------------------------------------------------------------------
    def _limpar_rollout(self):
        self.b_obs, self.b_acao, self.b_logp, self.b_valor = [], [], [], []
        self.b_rec, self.b_fim = [], []

    def ajustar_lr(self, fracao_restante):
        """Decai o lr linearmente: fracao_restante vai de 1 (início) até 0 (fim do treino)."""
        for grupo in self.otimizador.param_groups:
            grupo["lr"] = self.lr_inicial * fracao_restante

    # ------------------------------------------------------------------
    def amostrar_acao(self, estado):
        """Usada no treino: sorteia a ação da Gaussiana e devolve (ação, log π(a|s), V(s)).

        A ação devolvida NÃO é cortada em [-1, 1]: o log π é calculado sobre ela como foi
        sorteada. O ambiente é quem faz o clip (np.clip em EmbaixadinhaEnv.step).
        """
        with torch.no_grad():
            obs = torch.as_tensor(estado, dtype=torch.float32).unsqueeze(0)
            dist = self.rede.distribuicao(obs)
            acao = dist.sample()
            logp = dist.log_prob(acao).sum(-1)
            valor = self.rede.valor(obs)
        return acao.squeeze(0).numpy(), logp.item(), valor.item()

    def escolher_acao(self, estado, explorar=False):
        """Usada na avaliação: a média da Gaussiana (política determinística) ou um sorteio."""
        with torch.no_grad():
            obs = torch.as_tensor(estado, dtype=torch.float32).unsqueeze(0)
            dist = self.rede.distribuicao(obs)
            acao = dist.sample() if explorar else dist.mean
        return np.clip(acao.squeeze(0).numpy(), -1.0, 1.0)

    # ------------------------------------------------------------------
    def registrar(self, estado, acao, logp, valor, recompensa, terminou, truncou, proximo_estado):
        """Guarda um passo do rollout."""
        fim = terminou or truncou
        if self.norm_rec is not None:
            recompensa = self.norm_rec(recompensa, fim)

        # Truncar (limite de tempo) não é o fim de verdade: a bola ainda estava no ar.
        # Então o valor futuro é somado à recompensa antes de cortar o episódio
        # (mesma ideia do comentário sobre 'truncou' no treino do DQN).
        if truncou and not terminou:
            with torch.no_grad():
                v_prox = self.rede.valor(
                    torch.as_tensor(proximo_estado, dtype=torch.float32).unsqueeze(0)).item()
            recompensa += self.gamma * v_prox

        self.b_obs.append(estado)
        self.b_acao.append(acao)
        self.b_logp.append(logp)
        self.b_valor.append(valor)
        self.b_rec.append(recompensa)
        self.b_fim.append(float(fim))

    def rollout_pronto(self):
        return len(self.b_rec) >= self.passos_rollout

    # ------------------------------------------------------------------
    def _calcular_gae(self, ultimo_valor):
        """Vantagens por GAE(γ, λ). Percorre o rollout de trás para frente.

        δ_t = r_t + γ · V(s_{t+1}) · (1 - fim_t) - V(s_t)
        A_t = δ_t + γ · λ · (1 - fim_t) · A_{t+1}
        """
        n = len(self.b_rec)
        valores = np.array(self.b_valor, dtype=np.float32)
        recompensas = np.array(self.b_rec, dtype=np.float32)
        fins = np.array(self.b_fim, dtype=np.float32)

        vantagens = np.zeros(n, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(n)):
            proximo_valor = ultimo_valor if t == n - 1 else valores[t + 1]
            nao_fim = 1.0 - fins[t]
            delta = recompensas[t] + self.gamma * proximo_valor * nao_fim - valores[t]
            gae = delta + self.gamma * self.lam * nao_fim * gae
            vantagens[t] = gae
        retornos = vantagens + valores          # alvo do crítico
        return vantagens, retornos

    def atualizar(self, ultimo_estado):
        """Faz a atualização do PPO com o rollout acumulado e limpa o buffer.

        'ultimo_estado' é o estado seguinte ao último passo guardado (usado para
        completar o retorno quando o rollout termina no meio de um episódio).
        """
        with torch.no_grad():
            ultimo_valor = self.rede.valor(
                torch.as_tensor(ultimo_estado, dtype=torch.float32).unsqueeze(0)).item()
        vantagens, retornos = self._calcular_gae(ultimo_valor)

        obs = torch.as_tensor(np.array(self.b_obs), dtype=torch.float32)
        acoes = torch.as_tensor(np.array(self.b_acao), dtype=torch.float32)
        logp_antigo = torch.as_tensor(self.b_logp, dtype=torch.float32)
        vantagens = torch.as_tensor(vantagens)
        retornos = torch.as_tensor(retornos)
        n = len(obs)

        stats = {"perda_politica": [], "perda_valor": [], "entropia": [],
                 "kl": [], "frac_clip": []}
        for _ in range(self.epocas):
            ordem = torch.randperm(n)
            for ini in range(0, n, self.tamanho_lote):
                idx = ordem[ini:ini + self.tamanho_lote]

                dist = self.rede.distribuicao(obs[idx])
                logp = dist.log_prob(acoes[idx]).sum(-1)
                entropia = dist.entropy().sum(-1).mean()
                valor = self.rede.valor(obs[idx])

                # Vantagem normalizada dentro do minibatch
                adv = vantagens[idx]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                # Objetivo cortado do PPO
                log_razao = logp - logp_antigo[idx]
                razao = log_razao.exp()
                perda_politica = -torch.min(
                    razao * adv,
                    razao.clamp(1 - self.clip, 1 + self.clip) * adv).mean()

                perda_valor = 0.5 * (valor - retornos[idx]).pow(2).mean()
                perda = (perda_politica
                         + self.coef_valor * perda_valor
                         - self.coef_entropia * entropia)

                self.otimizador.zero_grad()
                perda.backward()
                nn.utils.clip_grad_norm_(self.rede.parameters(), self.max_grad_norm)
                self.otimizador.step()

                with torch.no_grad():
                    stats["perda_politica"].append(perda_politica.item())
                    stats["perda_valor"].append(perda_valor.item())
                    stats["entropia"].append(entropia.item())
                    stats["kl"].append(((razao - 1) - log_razao).mean().item())
                    stats["frac_clip"].append(((razao - 1).abs() > self.clip).float().mean().item())

        self._limpar_rollout()
        return {k: float(np.mean(v)) for k, v in stats.items()}

    # ------------------------------------------------------------------
    def salvar(self, caminho):
        torch.save(self.rede.state_dict(), caminho)

    def carregar(self, caminho):
        self.rede.load_state_dict(torch.load(caminho))
