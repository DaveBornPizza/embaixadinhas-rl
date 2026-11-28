"""
Treina o PPO (ações contínuas) no ambiente de embaixadinhas e gera métricas e gráficos.

Uso:
    python -m src.algoritmos.ppo.treino                        # 1.000.000 de passos
    python -m src.algoritmos.ppo.treino --passos 200000 --semente 1
    python -m src.algoritmos.ppo.treino --passos-como-dqn resultados/dqn/metricas.csv
        # usa como nº de passos a soma dos passos de todos os episódios do DQN (mesmo nº de amostras)

Saídas (pasta resultados/ppo/):
    metricas.csv   recompensa, embaixadinhas, passos e (preso=1 se acabou por bola presa) de cada episódio
    modelo.pt      pesos da rede treinada (ator + crítico)
    curvas.png     gráficos para os slides

Obs.: o treino é contado em PASSOS (decisões do agente), não em episódios, porque o PPO
atualiza a cada `passos_rollout` passos, não ao fim de cada episódio.
"""
import argparse
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                     # gera PNG sem abrir janela
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.env.ambiente import EmbaixadinhaEnv
from src.algoritmos.ppo.agente import AgentePPO

# Sobe 3 níveis: ppo/ -> algoritmos/ -> src/ -> embaixadinhas-rl/
RAIZ_PROJETO = Path(__file__).resolve().parents[3]
DIR_RESULTADOS = RAIZ_PROJETO / "resultados" / "ppo"


def media_movel(valores, janela=20):
    valores = np.asarray(valores, dtype=float)
    if len(valores) < janela:
        return valores
    kernel = np.ones(janela) / janela
    return np.convolve(valores, kernel, mode="valid")


def avaliar(env, escolher, n_episodios):
    """Roda n episódios com a função 'escolher(obs) -> ação' e devolve as embaixadinhas de cada um."""
    resultados = []
    for _ in range(n_episodios):
        obs, _ = env.reset()
        while True:
            obs, _, terminou, truncou, info = env.step(escolher(obs))
            if terminou or truncou:
                break
        resultados.append(info["embaixadinhas"])
    return np.array(resultados)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--passos", type=int, default=1_000_000, help="total de passos de treino")
    parser.add_argument("--passos-como-dqn", type=Path, default=None,
                        help="metricas.csv do DQN: define --passos como a soma da coluna 'passos'")
    parser.add_argument("--sem-anti-preso", action="store_true",
                        help="desliga a penalidade/término por bola presa na perna")
    parser.add_argument("--controle", choices=["torque", "velocidade"], default="torque",
                        help="como a ação contínua vira comando do motor (veja o ambiente)")
    parser.add_argument("--suavidade", type=float, default=0.02,
                        help="peso da penalidade por mudar a ação de um passo para o outro")
    parser.add_argument("--filtro", type=float, default=0.0,
                        help="filtro passa-baixa na ação, de 0 (desligado) a <1 (mais suave)")
    parser.add_argument("--log-std-inicial", type=float, default=-1.0)
    parser.add_argument("--semente", type=int, default=0)
    parser.add_argument("--saida", type=Path, default=DIR_RESULTADOS)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--passos-rollout", type=int, default=2048)
    parser.add_argument("--epocas", type=int, default=10)
    parser.add_argument("--lote", type=int, default=64)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--entropia", type=float, default=0.0)
    args = parser.parse_args()
    args.saida.mkdir(parents=True, exist_ok=True)

    if args.passos_como_dqn is not None:
        dqn = np.loadtxt(args.passos_como_dqn, delimiter=",", skiprows=1)
        args.passos = int(dqn[:, 2].sum())
        print(f"DQN: {len(dqn)} episódios = {args.passos} passos -> PPO treinará {args.passos} passos")

    # Sementes fixas: o resultado pode ser repetido
    random.seed(args.semente)
    np.random.seed(args.semente)
    torch.manual_seed(args.semente)

    env = EmbaixadinhaEnv(semente=args.semente, acao_continua=True,
                          anti_preso=not args.sem_anti_preso,
                          controle_continuo=args.controle, k_suavidade=args.suavidade,
                          alfa_filtro=args.filtro)
    agente = AgentePPO(env.dim_obs, env.n_acoes, lr=args.lr, clip=args.clip,
                       epocas=args.epocas, tamanho_lote=args.lote,
                       passos_rollout=args.passos_rollout, coef_entropia=args.entropia,
                       log_std_inicial=args.log_std_inicial)

    recompensas, toques, duracoes, presos = [], [], [], []
    obs, _ = env.reset()
    total_ep = 0.0
    passos = 0
    atualizacao = 0

    while passos < args.passos:
        agente.ajustar_lr(1.0 - passos / args.passos)

        # --- 1) Coleta um rollout --------------------------------------
        while not agente.rollout_pronto():
            acao, logp, valor = agente.amostrar_acao(obs)
            proximo, r, terminou, truncou, info = env.step(acao)

            # 'terminou' (bola caiu) corta o futuro; 'truncou' (limite de tempo) não.
            agente.registrar(obs, acao, logp, valor, r, terminou, truncou, proximo)
            passos += 1
            total_ep += r

            if terminou or truncou:
                recompensas.append(total_ep)
                toques.append(info["embaixadinhas"])
                duracoes.append(env.passo)
                presos.append(float(info["preso"]))
                total_ep = 0.0
                obs, _ = env.reset()
            else:
                obs = proximo

        # --- 2) Atualiza a política com o rollout ----------------------
        stats = agente.atualizar(obs)
        atualizacao += 1

        n = len(toques)
        if n:
            ultimos = slice(-20, None)
            print(f"passo {passos:8d} | ep {n:5d} | recompensa {np.mean(recompensas[ultimos]):8.2f} | "
                  f"embaixadinhas {np.mean(toques[ultimos]):6.2f} | "
                  f"presos {np.mean(presos[ultimos]):4.0%} | "
                  f"entropia {stats['entropia']:5.2f} | kl {stats['kl']:.4f} | "
                  f"clip {stats['frac_clip']:.2f} | L_valor {stats['perda_valor']:.3f}", flush=True)
        if atualizacao % 10 == 0:
            agente.salvar(args.saida / "modelo.pt")        # checkpoint

    # --- Salva métricas e modelo -------------------------------------
    dados = np.column_stack([recompensas, toques, duracoes, presos])
    np.savetxt(args.saida / "metricas.csv", dados, delimiter=",",
               header="recompensa,embaixadinhas,passos,preso", comments="")
    agente.salvar(args.saida / "modelo.pt")

    # --- Comparação final: política treinada x política aleatória ----
    aleatoria = avaliar(env, lambda o: np.random.uniform(-1, 1, env.n_acoes), 100)
    treinada = avaliar(env, lambda o: agente.escolher_acao(o, explorar=False), 100)
    print(f"\nEmbaixadinhas por episódio (100 episódios cada):")
    print(f"  aleatória: média {aleatoria.mean():.2f}, máximo {aleatoria.max()}")
    print(f"  treinada : média {treinada.mean():.2f}, máximo {treinada.max()}")

    # --- Gráficos -----------------------------------------------------
    fig, eixos = plt.subplots(1, 2, figsize=(12, 4.5))
    for eixo, valores, titulo, ref in (
        (eixos[0], recompensas, "Recompensa total por episódio", None),
        (eixos[1], toques, "Embaixadinhas por episódio", aleatoria.mean()),
    ):
        eixo.plot(valores, alpha=0.25, label="episódio")
        suave = media_movel(valores)
        eixo.plot(np.arange(len(suave)) + 19, suave, linewidth=2, label="média móvel (20)")
        if ref is not None:
            eixo.axhline(ref, color="gray", linestyle="--", label="política aleatória")
        eixo.set_title(titulo)
        eixo.set_xlabel("episódio")
        eixo.legend()
        eixo.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.saida / "curvas.png", dpi=150)
    print(f"\nResultados salvos em '{args.saida}/'")


if __name__ == "__main__":
    main()
