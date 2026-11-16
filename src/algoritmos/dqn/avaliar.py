"""
Mostra o agente treinado jogando.

Uso:
    python avaliar.py                    # abre uma janela e roda 5 episódios
    python avaliar.py --gif demo.gif     # grava um GIF (precisa de: pip install imageio)
    python avaliar.py --aleatorio        # mostra uma política aleatória, para comparar
"""
import argparse
import random

import numpy as np

from src.env.ambiente import EmbaixadinhaEnv
from src.algoritmos.dqn.agente import AgenteDQN

from pathlib import Path

# Sobe 3 níveis: dqn/ -> algoritmos/ -> src/ -> embaixadinhas-rl/
RAIZ_PROJETO = Path(__file__).resolve().parents[3]
DIR_RESULTADOS = RAIZ_PROJETO / "resultados" / "dqn"
DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo", default=DIR_RESULTADOS / "modelo.pt")
    parser.add_argument("--episodios", type=int, default=5)
    parser.add_argument("--gif", default=None, help="caminho do GIF de saída")
    parser.add_argument("--aleatorio", action="store_true")
    args = parser.parse_args()

    env = EmbaixadinhaEnv(render_mode="rgb_array" if args.gif else "human")
    agente = AgenteDQN(env.dim_obs, env.n_acoes)
    if not args.aleatorio:
        agente.carregar(args.modelo)

    quadros = []
    for ep in range(args.episodios):
        obs, _ = env.reset()
        while True:
            if args.aleatorio:
                acao = random.randrange(env.n_acoes)
            else:
                acao = agente.escolher_acao(obs, explorar=False)
            obs, _, terminou, truncou, info = env.step(acao)
            if args.gif:
                quadros.append(env.render())      # um quadro por decisão do agente
            if terminou or truncou:
                break
        print(f"episódio {ep + 1}: {info['embaixadinhas']} embaixadinhas")

    if args.gif:
        import imageio
        caminho_gif = DIR_RESULTADOS / args.gif
        imageio.mimsave(caminho_gif, quadros, fps=15)
        print(f"GIF salvo em {caminho_gif}")
    env.close()


if __name__ == "__main__":
    main()
