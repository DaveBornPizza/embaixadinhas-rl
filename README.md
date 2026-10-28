# Embaixadinhas com Aprendizado por Reforço

**Dupla:** Pedro Luyd Ribeiro Gonçalves e Gustavo Martins Medeiros

**Disciplina:** Inteligência Artifial, 5° Período

Este projeto desenvolve um simulador 2D de embaixadinhas e treina um agente por Aprendizado por Reforço para controlar uma perna com três articulações (quadril, joelho e tornozelo). O objetivo é maximizar o número de embaixadinhas consecutivas sem deixar a bola cair, usando ações contínuas de torque e recompensa baseada nos toques na bola.

## 1. Ambiente e estado markoviano

**Ambiente.** Simulação 2D (plano sagital) com gravidade. Uma perna com quadril fixo no espaço e 3 articulações: quadril (move a coxa), joelho e tornozelo (move o pé). Uma bola começa caindo de uma posição inicial sorteada na parte superior da tela.

**Estado observado** (vetor com 10 variáveis):

| Grupo | Variáveis |
|---|---|
| Bola | posição (x, y) e velocidade (vx, vy) |
| Articulações | ângulos (θ₁, θ₂, θ₃) |
| Articulações | velocidades angulares (ω₁, ω₂, ω₃) |

**Por que é markoviano.** Posições e velocidades de todos os corpos bastam para prever o próximo estado a partir da dinâmica, sem depender do histórico.

**Fim do episódio.** A bola toca o chão, sai da área de alcance ou o limite de passos é atingido.

## 2. Ações do agente

Espaço de ações **contínuo**: `a = (τ₁, τ₂, τ₃) ∈ [-1, 1]³`, um torque normalizado por articulação a cada passo de tempo (quadril, joelho, tornozelo).

No Gymnasium: `Box(low=-1, high=1, shape=(3,))`.

Justificativa: tocar a bola exige controle fino de força e ângulo. Algoritmos candidatos: SAC, PPO e TD3.

## 3. Recompensas

| Evento | Recompensa |
|---|---|
| Embaixadinha (toque que lança a bola para cima) | +10 |
| Bola no ar, por passo de tempo | +0,01 |
| Distância horizontal entre pé e bola (shaping) | −0,1 · \|Δx\| |
| Esforço dos motores | −0,001 · ‖a‖² |
| Bola toca o chão (fim do episódio) | −10 |

Os valores são iniciais e serão calibrados nos experimentos. Uma embaixadinha é contada quando há contato pé-bola e a velocidade vertical da bola passa de negativa para positiva.

## Estrutura do repositório

```
.
├── README.md
├── apresentacao/      # slides da entrega 1 (PDF/PPTX)
├── src/               # ambiente e agentes (entregas futuras)
└── requirements.txt
```

## Como executar

_A preencher nas próximas entregas._

## Entregas

- [x] Entrega 1: descrição do problema (apresentação em 29/10/2026)
- [ ] Entrega 2: [a definir]
