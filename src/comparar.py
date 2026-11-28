"""
Compara DQN e PPO com o MESMO protocolo de teste (sem janela, só números e gráficos).

Uso:
    python -m src.comparar                         # 3 sementes x 200 episódios (600 por agente)
    python -m src.comparar --episodios-por-semente 50    # versão rápida, para testar
    python -m src.comparar --regras original       # só a regra original (mais rápido)

O que faz:
  * Política gulosa/determinística (sem exploração) para DQN e PPO, 3 sementes de ambiente.
  * Linhas de base: perna parada (velocidade-alvo 0, como no README do DQN) e política aleatória.
  * Cada agente é testado em DUAS versões do ambiente:
      - "original": a regra de bola presa DESLIGADA (o jogo do DQN, como na entrega 2);
      - "preso":    a regra de bola presa LIGADA (a regra usada no treino do PPO).
    Assim a comparação é justa nos dois sentidos.
  * Mede também a "agitação" da perna: média de sum(|Δω|) por decisão, em rad/s. Valor menor =
    movimento mais suave (é uma medida objetiva do "flicker").
  * Gráfico de aprendizado dos dois no MESMO eixo (passos de ambiente), lido dos metricas.csv.

Saídas (pasta resultados/comparacao/):
    resumo.csv, episodios.csv, comparacao.png, curvas_comparacao.png
"""
import argparse
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.env.ambiente import EmbaixadinhaEnv, ACOES
from src.algoritmos.dqn.agente import AgenteDQN
from src.algoritmos.ppo.agente import AgentePPO

RAIZ = Path(__file__).resolve().parents[1]
DIR_DQN = RAIZ / "resultados" / "dqn"
DIR_PPO = RAIZ / "resultados" / "ppo"
DIR_SAIDA = RAIZ / "resultados" / "comparacao"

ACAO_PARADA = int(np.argmin(np.abs(ACOES).sum(axis=1)))   # índice da ação (0, 0, 0): perna parada


def criar_env(continua, anti_preso, semente):
    return EmbaixadinhaEnv(semente=semente, acao_continua=continua, anti_preso=anti_preso,
                           controle_continuo="torque")


def rodar(env, escolher, n_episodios):
    """Roda n episódios. Devolve um dict com um vetor por métrica (um valor por episódio)."""
    toques, passos, completou, preso, agitacao = [], [], [], [], []
    for _ in range(n_episodios):
        obs, _ = env.reset()
        w_ant = np.array([j.speed for j in env.juntas])
        soma, n = 0.0, 0
        while True:
            obs, _, terminou, truncou, info = env.step(escolher(obs))
            w = np.array([j.speed for j in env.juntas])
            soma += float(np.abs(w - w_ant).sum())
            w_ant = w
            n += 1
            if terminou or truncou:
                break
        toques.append(info["embaixadinhas"])
        passos.append(env.passo)
        completou.append(float(truncou))          # chegou ao limite de tempo sem terminar
        preso.append(float(info["preso"]))
        agitacao.append(soma / max(n, 1))
    return {"toques": np.array(toques, dtype=float), "passos": np.array(passos, dtype=float),
            "completou": np.array(completou), "preso": np.array(preso),
            "agitacao": np.array(agitacao)}


def juntar(lista):
    return {k: np.concatenate([d[k] for d in lista]) for k in lista[0]}


def avaliar_agente(nome, fabricar_escolha, continua, anti_preso, sementes, n):
    partes = []
    for s in sementes:
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
        env = criar_env(continua, anti_preso, s)
        partes.append(rodar(env, fabricar_escolha(env), n))
        env.close()
    r = juntar(partes)
    print(f"  {nome:<26} média {r['toques'].mean():6.1f} | mediana {np.median(r['toques']):5.1f} | "
          f"máx {r['toques'].max():4.0f} | completos {r['completou'].mean():5.1%} | "
          f"presos {r['preso'].mean():5.1%} | agitação {r['agitacao'].mean():5.2f}", flush=True)
    return r


def fabrica_dqn(caminho):
    def f(env):
        agente = AgenteDQN(env.dim_obs, env.n_acoes)
        agente.carregar(caminho)
        return lambda o: agente.escolher_acao(o, explorar=False)
    return f


def fabrica_ppo(caminho):
    def f(env):
        agente = AgentePPO(env.dim_obs, env.n_acoes)
        agente.carregar(caminho)
        return lambda o: agente.escolher_acao(o, explorar=False)
    return f


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dqn", type=Path, default=DIR_DQN / "modelo.pt")
    parser.add_argument("--ppo", type=Path, default=DIR_PPO / "modelo.pt")
    parser.add_argument("--ppo-melhor", type=Path, default=DIR_PPO / "modelo_melhor.pt",
                        help="se o arquivo existir, também avalia esse checkpoint")
    parser.add_argument("--sementes", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--episodios-por-semente", type=int, default=200)
    parser.add_argument("--regras", choices=["ambas", "original", "preso"], default="ambas")
    args = parser.parse_args()

    DIR_SAIDA.mkdir(parents=True, exist_ok=True)
    n = args.episodios_por_semente
    total = n * len(args.sementes)
    regras = {"ambas": [("original", False), ("preso", True)],
              "original": [("original", False)], "preso": [("preso", True)]}[args.regras]

    agentes = [("DQN", fabrica_dqn(args.dqn), False), ("PPO", fabrica_ppo(args.ppo), True)]
    if args.ppo_melhor.exists():
        agentes.append(("PPO (melhor checkpoint)", fabrica_ppo(args.ppo_melhor), True))

    resultados = {}            # (nome, regra) -> métricas por episódio

    print(f"Linhas de base ({total} episódios, regra original):")
    resultados[("Perna parada", "original")] = avaliar_agente(
        "Perna parada", lambda env: (lambda o: ACAO_PARADA), False, False, args.sementes, n)
    resultados[("Aleatória", "original")] = avaliar_agente(
        "Aleatória", lambda env: (lambda o: random.randrange(env.n_acoes)),
        False, False, args.sementes, n)

    for nome_regra, anti_preso in regras:
        print(f"\nRegra '{nome_regra}' (anti_preso={anti_preso}), {total} episódios por agente:")
        for nome, fabrica, continua in agentes:
            resultados[(nome, nome_regra)] = avaliar_agente(
                nome, fabrica, continua, anti_preso, args.sementes, n)

    # --- Tabelas ------------------------------------------------------------
    with open(DIR_SAIDA / "resumo.csv", "w", encoding="utf-8") as f:
        f.write("agente,regra,episodios,media,mediana,maximo,completos,presos,agitacao\n")
        for (nome, regra), r in resultados.items():
            f.write(f"{nome},{regra},{len(r['toques'])},{r['toques'].mean():.2f},"
                    f"{np.median(r['toques']):.1f},{r['toques'].max():.0f},"
                    f"{r['completou'].mean():.4f},{r['preso'].mean():.4f},"
                    f"{r['agitacao'].mean():.3f}\n")
    with open(DIR_SAIDA / "episodios.csv", "w", encoding="utf-8") as f:
        f.write("agente,regra,embaixadinhas,passos,completou,preso,agitacao\n")
        for (nome, regra), r in resultados.items():
            for i in range(len(r["toques"])):
                f.write(f"{nome},{regra},{r['toques'][i]:.0f},{r['passos'][i]:.0f},"
                        f"{r['completou'][i]:.0f},{r['preso'][i]:.0f},{r['agitacao'][i]:.3f}\n")

    # --- Gráfico 1: embaixadinhas por episódio (regra original e/ou preso) ---------
    chaves = [("Perna parada", "original"), ("Aleatória", "original")]
    chaves += [(nome, regra) for regra, _ in regras for nome, _, _ in agentes]
    rotulos = [f"{nome}\n({regra})" if nome not in ("Perna parada", "Aleatória") else nome
               for nome, regra in chaves]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
    medias = [resultados[k]["toques"].mean() for k in chaves]
    barras = a1.bar(range(len(chaves)), medias, color="#2e7d32")
    a1.bar_label(barras, fmt="%.1f")
    a1.set_xticks(range(len(chaves)))
    a1.set_xticklabels(rotulos, fontsize=8)
    a1.set_title(f"Média de embaixadinhas por episódio ({total} episódios)")
    a1.grid(axis="y", alpha=0.3)
    a2.boxplot([resultados[k]["toques"] for k in chaves], showfliers=False)
    a2.set_xticklabels(rotulos, fontsize=8)
    a2.set_title("Distribuição por episódio")
    a2.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(DIR_SAIDA / "comparacao.png", dpi=150)
    plt.close(fig)

    # --- Gráfico 2: aprendizado dos dois no mesmo eixo (passos de ambiente) ---------
    def curva(caminho, janela=100):
        d = np.loadtxt(caminho, delimiter=",", skiprows=1)
        passos = np.cumsum(d[:, 2])
        suave = np.convolve(d[:, 1], np.ones(janela) / janela, mode="valid")
        return passos[janela - 1:], suave

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for rotulo, caminho, cor in (("DQN (ε-gulosa, 5% aleatória no fim)", DIR_DQN / "metricas.csv", "#1565c0"),
                                 ("PPO (política com ruído)", DIR_PPO / "metricas.csv", "#e65100")):
        if caminho.exists():
            x, y = curva(caminho)
            ax.plot(x / 1e6, y, label=rotulo, color=cor, linewidth=2)
    ax.set_xlabel("passos de ambiente (milhões)")
    ax.set_ylabel("embaixadinhas por episódio (média móvel de 100)")
    ax.set_title("Aprendizado durante o treino, mesmo orçamento de passos")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(DIR_SAIDA / "curvas_comparacao.png", dpi=150)
    plt.close(fig)

    print(f"\nSalvo em '{DIR_SAIDA}/': resumo.csv, episodios.csv, comparacao.png, curvas_comparacao.png")


if __name__ == "__main__":
    main()
