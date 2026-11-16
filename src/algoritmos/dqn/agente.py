"""
DQN (Deep Q-Network) escrito à mão com PyTorch.

Peças:
    RedeQ        rede neural que estima Q(s, a) para todas as ações de uma vez
    ReplayBuffer memória de transições (s, a, r, s', terminou)
    AgenteDQN    escolhe ações (epsilon-greedy) e atualiza a rede
"""
import collections
import random

import numpy as np
import torch
import torch.nn as nn


class RedeQ(nn.Module):
    """Entrada: estado (10 números). Saída: um valor Q para cada uma das 27 ações."""

    def __init__(self, dim_obs, n_acoes, oculta=128):
        super().__init__()
        self.rede = nn.Sequential(
            nn.Linear(dim_obs, oculta), nn.ReLU(),
            nn.Linear(oculta, oculta), nn.ReLU(),
            nn.Linear(oculta, n_acoes),
        )

    def forward(self, x):
        return self.rede(x)


class ReplayBuffer:
    """Guarda as últimas 'capacidade' transições e sorteia lotes para o treino."""

    def __init__(self, capacidade):
        self.memoria = collections.deque(maxlen=capacidade)

    def guardar(self, estado, acao, recompensa, proximo_estado, terminou):
        self.memoria.append((estado, acao, recompensa, proximo_estado, terminou))

    def amostrar(self, tamanho_lote):
        lote = random.sample(self.memoria, tamanho_lote)
        estados, acoes, recompensas, proximos, terminos = zip(*lote)
        return (
            torch.as_tensor(np.array(estados), dtype=torch.float32),
            torch.as_tensor(acoes, dtype=torch.int64),
            torch.as_tensor(recompensas, dtype=torch.float32),
            torch.as_tensor(np.array(proximos), dtype=torch.float32),
            torch.as_tensor(terminos, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.memoria)


class AgenteDQN:
    def __init__(self, dim_obs, n_acoes,
                 gamma=0.99,              # desconto: quanto vale a recompensa futura
                 lr=1e-3,                 # taxa de aprendizado do otimizador
                 tamanho_lote=64,
                 capacidade_buffer=100_000,
                 passos_inicio=1_000,     # só treina depois de juntar essa quantidade de transições
                 intervalo_alvo=500,      # a cada quantos passos a rede-alvo é sincronizada
                 eps_inicial=1.0, eps_final=0.05,
                 eps_passos=30_000):      # em quantos passos o epsilon cai de inicial a final
        self.n_acoes = n_acoes
        self.gamma = gamma
        self.tamanho_lote = tamanho_lote
        self.passos_inicio = passos_inicio
        self.intervalo_alvo = intervalo_alvo
        self.eps_inicial, self.eps_final, self.eps_passos = eps_inicial, eps_final, eps_passos

        self.rede = RedeQ(dim_obs, n_acoes)          # rede que está sendo treinada
        self.rede_alvo = RedeQ(dim_obs, n_acoes)     # cópia "congelada" usada nos alvos
        self.rede_alvo.load_state_dict(self.rede.state_dict())
        self.otimizador = torch.optim.Adam(self.rede.parameters(), lr=lr)
        self.buffer = ReplayBuffer(capacidade_buffer)
        self.passos = 0                              # total de passos de simulação até agora

    # ------------------------------------------------------------------
    def epsilon(self):
        """Cai linearmente de eps_inicial até eps_final e depois fica constante."""
        fracao = min(1.0, self.passos / self.eps_passos)
        return self.eps_inicial + fracao * (self.eps_final - self.eps_inicial)

    def escolher_acao(self, estado, explorar=True):
        """Epsilon-greedy: com probabilidade epsilon sorteia; senão escolhe o maior Q."""
        if explorar and random.random() < self.epsilon():
            return random.randrange(self.n_acoes)
        with torch.no_grad():
            q = self.rede(torch.as_tensor(estado, dtype=torch.float32).unsqueeze(0))
        return int(q.argmax(dim=1).item())

    def registrar(self, estado, acao, recompensa, proximo_estado, terminou):
        self.buffer.guardar(estado, acao, recompensa, proximo_estado, terminou)
        self.passos += 1

    def atualizar(self):
        """Um passo de gradiente. Devolve a perda (ou None se ainda é cedo)."""
        if len(self.buffer) < max(self.passos_inicio, self.tamanho_lote):
            return None

        estados, acoes, recompensas, proximos, terminos = self.buffer.amostrar(self.tamanho_lote)

        # Q(s, a) previsto pela rede, só para a ação que de fato foi tomada
        q_previsto = self.rede(estados).gather(1, acoes.unsqueeze(1)).squeeze(1)

        # Alvo de Bellman: r + gamma * max_a' Q_alvo(s', a'), sem o 2º termo se o episódio terminou
        with torch.no_grad():
            q_proximo = self.rede_alvo(proximos).max(dim=1).values
            alvo = recompensas + self.gamma * q_proximo * (1.0 - terminos)

        perda = nn.functional.smooth_l1_loss(q_previsto, alvo)   # Huber loss
        self.otimizador.zero_grad()
        perda.backward()
        nn.utils.clip_grad_norm_(self.rede.parameters(), 10.0)   # evita gradientes enormes
        self.otimizador.step()

        if self.passos % self.intervalo_alvo == 0:
            self.rede_alvo.load_state_dict(self.rede.state_dict())
        return perda.item()

    def salvar(self, caminho):
        torch.save(self.rede.state_dict(), caminho)

    def carregar(self, caminho):
        self.rede.load_state_dict(torch.load(caminho))
        self.rede_alvo.load_state_dict(self.rede.state_dict())
