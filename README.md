# Embaixadinhas com Aprendizado por Reforço

**Dupla:** Pedro Luyd Ribeiro Gonçalves e Gustavo Martins Medeiros

**Disciplina:** Inteligência Artificial, 5° Período

Este projeto desenvolve um simulador 2D de embaixadinhas e treina um agente por Aprendizado por Reforço para controlar uma perna com três articulações (quadril, joelho e tornozelo). O objetivo é maximizar o número de embaixadinhas consecutivas sem deixar a bola cair, usando ações contínuas de torque e recompensa baseada nos toques na bola.

## 1. Ambiente e estado markoviano

**Ambiente:** Simulação 2D (plano sagital) com gravidade. Uma perna com quadril fixo no espaço e 3 articulações: quadril (move a coxa), joelho e tornozelo (move o pé). Uma bola começa caindo de uma posição inicial sorteada na parte superior da tela.

**Estado observado** (vetor com 10 variáveis):

| Grupo | Variáveis |
|---|---|
| Bola | posição (x, y) e velocidade (vx, vy) |
| Articulações | ângulos (θ₁, θ₂, θ₃) |
| Articulações | velocidades angulares (ω₁, ω₂, ω₃) |

**Por que é markoviano:** Posições e velocidades de todos os corpos bastam para prever o próximo estado a partir da dinâmica, sem depender do histórico.

**Fim do episódio:** A bola toca o chão, sai da área de alcance ou o limite de passos é atingido.

## 2. Ações do agente

O ambiente (`EmbaixadinhaEnv`) suporta dois modos de controle, selecionados pelo parâmetro `acao_continua`:

### Espaço de ações **contínuo** (`acao_continua=True`):
`a = (τ₁, τ₂, τ₃) ∈ [-1, 1]³`, um torque normalizado por articulação a cada passo de tempo (quadril, joelho, tornozelo), escalado até `TORQUE_MAX` (80 N·m).

Justificativa: tocar a bola exige controle fino de força e ângulo. Algoritmos candidatos: SAC, PPO e TD3.

### Espaço de ações **discreto** (`acao_continua=False` — usado no DQN):
um inteiro de 0 a 26. Cada articulação (quadril, joelho, tornozelo) recebe um nível em {-1, 0, +1}, o que dá 3 × 3 × 3 = 27 combinações. A tabela `ACOES` (formato 27 × 3) converte o inteiro escolhido nos 3 níveis, que definem a **velocidade-alvo do motor** da articulação (`nível × VEL_MAX`). O nível 0 mantém a articulação parada, segurando a pose.

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
├── requirements.txt
├── apresentacao/              # Slides e materiais das entregas (PDF/PPTX)
├── resultados/                # Modelos salvos, métricas e gráficos organizados por algoritmo
│   └── dqn/
└── src/
    ├── env/
    │   └── ambiente.py        # Ambiente de simulação compartilhado entre os agentes
    └── algoritmos/            # Implementações dos algoritmos de RL
        └── dqn/               # Agente, script de treino e avaliação do DQN
```

## Como executar

```bash
pip install swig              # necessário para compilar o Box2D
pip install -r requirements.txt
python -m src.algoritmos.dqn.treino --episodios 1000   # treina e salva em resultados/dqn/
python -m src.algoritmos.dqn.avaliar                   # abre a janela com o agente jogando
python -m src.algoritmos.dqn.avaliar --gif demo.gif    # ou grava um GIF
```
### Parâmetros do ambiente

Parâmetros do ambiente (em `src/env/ambiente.py`): física a 60 Hz com `FRAME_SKIP = 4` (cada decisão dura 1/15 s), episódio de até 500 decisões, perna com densidade 15 e torque máximo de motor 80 N.m, bola leve (densidade 0,3) com restituição 0,5.

## Estado atual [ATUALIZAR]

- Ambiente (Box2D + pygame): funcionando. Perna parada rende cerca de 2 embaixadinhas por episódio (a bola quica passivamente no pé), e a política aleatória rende em média cerca de 0,2.
- Regras de colisão: a perna é sólida (a bola bate na coxa e na canela), mas o **tornozelo só pode abrir**, nunca fechar o "L" do pé contra a canela, o que impediria prender a bola. Em uma varredura de 729 padrões de ação com a bola começando no canto do "L", 8 ainda carregaram a bola sobre o pé quase o tempo todo (sem embaixadinhas), então um agente pode aprender a "carregar" a bola em vez de quicar. A constante `BOLA_COLIDE_PERNA = False` faz a bola atravessar a perna.
- Fim de episódio: o contato da bola com o chão é detectado pelo Box2D durante a simulação, no primeiro toque (e não só no fim de cada decisão).
- Contagem de embaixadinha: o pé toca a bola, a bola sai subindo a mais de 1 m/s, estava acima do pé e passaram pelo menos 12 passos de física (0,2 s) desde a contagem anterior.
- Treino com DQN: **ainda não aprende**. Em um teste de 1500 episódios (antes das correções de colisão) a recompensa ficou estável em torno de -7, e a política treinada fez 0 embaixadinhas. Causa em investigação (hipóteses: peso do termo de distância, tamanho do buffer, possível erro na implementação).
- `avaliar.py` não foi testado com janela real nem com GIF.


## Entregas

- [x] Entrega 1: descrição do problema (apresentação em 29/10/2026)
- [x] Entrega 2: implementação do problema usando o algoritmo do Deep Q-Network (apresentação em 17/11/2026)
- [ ] Entrega 3: [a definir]

## Créditos
 
- Imagem do esquema da perna (quadril, joelho e tornozelo) usada no slide 2: H. Lv e B. Hao, *Convolutional neural network and long short-term memory hybrid model-based gait prediction for spacesuit intelligent assistive device in low gravity environment*.
 
