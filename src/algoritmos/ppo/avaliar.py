"""
Mostra o agente PPO treinado jogando.

Uso:
    python -m src.algoritmos.ppo.avaliar                    # abre uma janela e roda 5 episódios
    python -m src.algoritmos.ppo.avaliar --gif demo.gif     # grava um GIF (precisa de: pip install imageio)
    python -m src.algoritmos.ppo.avaliar --aleatorio        # mostra uma política aleatória, para comparar
    python -m src.algoritmos.ppo.avaliar --estocastica      # sorteia da Gaussiana em vez de usar a média
"""
import argparse
from pathlib import Path

import numpy as np

from src.env.ambiente import EmbaixadinhaEnv
from src.algoritmos.ppo.agente import AgentePPO

# Sobe 3 níveis: ppo/ -> algoritmos/ -> src/ -> embaixadinhas-rl/
RAIZ_PROJETO = Path(__file__).resolve().parents[3]
DIR_RESULTADOS = RAIZ_PROJETO / "resultados" / "ppo"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo", default=DIR_RESULTADOS / "modelo.pt")
    parser.add_argument("--episodios", type=int, default=5)
    parser.add_argument("--gif", default=None, help="nome do GIF de saída (salvo em resultados/ppo/)")
    parser.add_argument("--aleatorio", action="store_true")
    parser.add_argument("--estocastica", action="store_true",
                        help="usa a política com exploração (padrão: média da Gaussiana)")
    parser.add_argument("--controle", choices=["torque", "velocidade"], default="torque",
                        help="precisa ser o mesmo usado no treino")
    parser.add_argument("--filtro", type=float, default=0.0,
                        help="precisa ser o mesmo usado no treino")
    parser.add_argument("--sem-anti-preso", action="store_true",
                        help="não encerra o episódio quando a bola fica presa na perna")
    args = parser.parse_args()

    env = EmbaixadinhaEnv(render_mode="rgb_array" if args.gif else "human", acao_continua=True,
                         anti_preso=not args.sem_anti_preso,
                         controle_continuo=args.controle, alfa_filtro=args.filtro)
    agente = AgentePPO(env.dim_obs, env.n_acoes)
    if not args.aleatorio:
        agente.carregar(args.modelo)

    quadros = []
    for ep in range(args.episodios):
        obs, _ = env.reset()
        while True:
            if args.aleatorio:
                acao = np.random.uniform(-1, 1, env.n_acoes)
            else:
                acao = agente.escolher_acao(obs, explorar=args.estocastica)
            obs, _, terminou, truncou, info = env.step(acao)
            if args.gif:
                quadros.append(env.render())      # um quadro por decisão do agente
            if terminou or truncou:
                break
        print(f"episódio {ep + 1}: {info['embaixadinhas']} embaixadinhas"
              + (" (terminou por bola presa)" if info["preso"] else ""))

    if args.gif:
        import imageio
        DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
        caminho_gif = DIR_RESULTADOS / args.gif
        imageio.mimsave(caminho_gif, quadros, fps=15)
        print(f"GIF salvo em {caminho_gif}")
    env.close()


if __name__ == "__main__":
    main()
