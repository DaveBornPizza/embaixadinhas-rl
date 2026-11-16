"""
Treina o DQN no ambiente de embaixadinhas e gera métricas e gráficos.

Uso:
    python treino.py                         # 1000 episódios
    python treino.py --episodios 300 --semente 1

Saídas (pasta resultados/):
    metricas.csv   recompensa, embaixadinhas e passos de cada episódio
    modelo.pt      pesos da rede treinada
    curvas.png     gráficos para os slides
"""
import argparse
import os
import random

import matplotlib
matplotlib.use("Agg")                     # gera PNG sem abrir janela
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.env.ambiente import EmbaixadinhaEnv
from src.algoritmos.dqn.agente import AgenteDQN

from pathlib import Path

# Sobe 3 níveis: dqn/ -> algoritmos/ -> src/ -> embaixadinhas-rl/
RAIZ_PROJETO = Path(__file__).resolve().parents[3]
DIR_RESULTADOS = RAIZ_PROJETO / "resultados" / "dqn"
DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)


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
    parser.add_argument("--episodios", type=int, default=1000)
    parser.add_argument("--semente", type=int, default=0)
    parser.add_argument("--saida", type=Path, default=DIR_RESULTADOS)
    args = parser.parse_args()
    args.saida.mkdir(parents=True, exist_ok=True)

    # Sementes fixas: o resultado pode ser repetido
    random.seed(args.semente)
    np.random.seed(args.semente)
    torch.manual_seed(args.semente)
    os.makedirs(args.saida, exist_ok=True)

    env = EmbaixadinhaEnv(semente=args.semente)
    agente = AgenteDQN(env.dim_obs, env.n_acoes)

    recompensas, toques, duracoes = [], [], []
    for ep in range(1, args.episodios + 1):
        obs, _ = env.reset()
        total = 0.0
        while True:
            acao = agente.escolher_acao(obs)
            proximo, r, terminou, truncou, info = env.step(acao)

            # 'terminou' (bola caiu) corta o futuro. 'truncou' (limite de tempo) não:
            # a bola ainda poderia continuar no ar, então o valor futuro segue contando.
            agente.registrar(obs, acao, r, proximo, terminou)
            agente.atualizar()

            obs = proximo
            total += r
            if terminou or truncou:
                break

        recompensas.append(total)
        toques.append(info["embaixadinhas"])
        duracoes.append(env.passo)

        if ep % 20 == 0:
            print(f"ep {ep:4d} | recompensa {np.mean(recompensas[-20:]):7.2f} | "
                  f"embaixadinhas {np.mean(toques[-20:]):5.2f} | "
                  f"epsilon {agente.epsilon():.2f}", flush=True)

    # --- Salva métricas e modelo -------------------------------------
    dados = np.column_stack([recompensas, toques, duracoes])
    np.savetxt(args.saida / "metricas.csv", dados, delimiter=",",
           header="recompensa,embaixadinhas,passos", comments="")
    agente.salvar(args.saida / "modelo.pt")

    # --- Comparação final: política treinada x política aleatória ----
    aleatoria = avaliar(env, lambda o: random.randrange(env.n_acoes), 100)
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
