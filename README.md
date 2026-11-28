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

Implementação: o torque é aplicado pelo motor da junta do Box2D. O sinal de `a` define o sentido e `|a| · TORQUE_MAX` o torque máximo; a velocidade da junta fica limitada a `VEL_MAX` (3 rad/s), acima da qual o motor freia, o que funciona como amortecimento. Sem esse limitador a perna não tem nenhum atrito nas juntas. Existe também o modo `controle_continuo="velocidade"` (velocidade-alvo `a · VEL_MAX`, equivalente contínuo das 27 ações do DQN), usado apenas como comparação.

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

Termos extras usados **só no PPO** (desligados por padrão no ambiente, o DQN não os usa):

| Termo | Valor | Motivo |
|---|---|---|
| Bola presa na perna (`anti_preso`) | −0,5 por decisão após 0,25 s encostada; o episódio termina (−10) após 1 s; sem o +0,01 enquanto encostada | evitar que o agente carregue a bola parada no canto pé-canela só para ganhar a recompensa de sobrevivência |
| Variação da ação (`k_suavidade`) | −0,02 · ‖aₜ − aₜ₋₁‖² | pilotos de 300 mil passos (uma semente cada), avaliação determinística: com k = 0 todos os episódios terminaram com a bola presa (~12 embaixadinhas); com k = 0,02 nenhum terminou preso (até ~70). A perna pareceu piscar um pouco mais com k = 0,02. |

Os valores são iniciais e serão calibrados nos experimentos. Uma embaixadinha é contada quando o pé toca a bola, ela sai subindo a mais de 1 m/s, estava acima do pé e passaram pelo menos 0,2 s desde a contagem anterior.

## Estrutura do repositório

```
.
├── README.md
├── requirements.txt
├── apresentacao/              # Slides e materiais das entregas (PDF/PPTX)
├── resultados/                # Modelos salvos, métricas e gráficos organizados por algoritmo
│   ├── dqn/
│   └── ppo/
└── src/
    ├── comparar.py            # Teste padronizado (DQN x PPO x linhas de base) e gráficos da entrega 3
    ├── env/
    │   └── ambiente.py        # Ambiente de simulação compartilhado entre os agentes
    └── algoritmos/            # Implementações dos algoritmos de RL
        ├── dqn/               # Agente, script de treino e avaliação do DQN
        └── ppo/               # Agente, script de treino e avaliação do PPO (ações contínuas)
```

## Como executar

```bash
pip install swig              # necessário para compilar o Box2D
pip install -r requirements.txt
python -m src.algoritmos.dqn.treino --episodios 1000   # treina e salva em resultados/dqn/
python -m src.algoritmos.dqn.avaliar                   # abre a janela com o agente jogando
python -m src.algoritmos.dqn.avaliar --gif demo.gif    # ou grava um GIF

python -m src.algoritmos.ppo.treino --passos-como-dqn resultados/dqn/metricas.csv
                                                       # PPO (ação contínua) com o mesmo nº de passos do DQN;
                                                       # salva em resultados/ppo/ (use --saida para outra pasta)
python -m src.algoritmos.ppo.avaliar                   # abre a janela com o agente PPO jogando (política determinística)
python -m src.algoritmos.ppo.avaliar --estocastica     # idem, com o ruído da política
python -m src.algoritmos.ppo.avaliar --gif demo_ppo.gif --episodios 1   # grava um GIF em resultados/ppo/ (precisa de: pip install imageio)

python -m src.comparar                                 # DQN x PPO x linhas de base (600 episódios cada), em resultados/comparacao/
```
### Parâmetros do ambiente

Parâmetros do ambiente (em `src/env/ambiente.py`): física a 60 Hz com `FRAME_SKIP = 4` (cada decisão dura 1/15 s), episódio de até 500 decisões, perna com densidade 15 e torque máximo de motor 80 N.m, bola leve (densidade 0,3) com restituição 0,5.

## Estado atual

- Ambiente (Box2D + pygame): funcionando, com **ação contínua e discreta** (parâmetro `acao_continua`). A adaptação foi necessária porque o DQN só escolhe entre um conjunto finito de ações.
  - Discreto (padrão, usado no DQN): 27 ações, cada articulação recebe velocidade-alvo `nível × VEL_MAX`, com nível em {-1, 0, +1}.
  - Contínuo (usado no PPO): torque normalizado em [-1, 1] por articulação (o sinal dá o sentido do motor e o módulo a fração de `TORQUE_MAX`), com a velocidade da junta limitada a `VEL_MAX`.
- Regras de colisão: a perna é sólida (`BOLA_COLIDE_PERNA = True`: a bola bate na coxa e na canela), mas o **tornozelo só pode abrir** (limite de -0,6 a 0 rad), nunca fechar o "L" do pé contra a canela, o que impediria prender a bola. Em uma varredura de 729 padrões de ação com a bola começando no canto do "L", 8 ainda carregaram a bola sobre o pé quase o tempo todo (sem embaixadinhas), então um agente poderia aprender a "carregar" a bola em vez de quicar. Com `BOLA_COLIDE_PERNA = False` a bola atravessa a perna.
- Fim de episódio: o contato da bola com o chão é detectado pelo Box2D durante a simulação, no primeiro toque (e não só no fim de cada decisão).
- Contagem de embaixadinha: o pé toca a bola, a bola sai subindo a mais de 1 m/s, estava acima do pé e passaram pelo menos 12 passos de física (0,2 s) desde a contagem anterior.
- Treino com DQN: **aprende**. Treinado por 8000 episódios (`resultados/dqn/`). Até o episódio 2000 a média fica em ~0,5 embaixadinha por episódio; o salto acontece por volta do episódio 2500, e do 3000 em diante a média de treino fica em ~23 por episódio (ainda com 5% de ações aleatórias de exploração).
- Avaliação do `modelo.pt` (política gulosa, sem exploração, 600 episódios em 3 sementes):

  | Política | Embaixadinhas por episódio |
  |---|---|
  | Perna parada | 1,4 em média (mediana 2) |
  | Aleatória | 0,2 em média |
  | **Agente DQN** | **78,5 em média** (mediana 80, máximo 92) |

- ~94% dos episódios do agente chegam ao limite de 500 decisões (~33 s) sem a bola cair, então hoje o teto é o limite de tempo do episódio.
- `avaliar.py`: gravação de GIF testada e funcionando; modo janela ainda não conferido.



### PPO (ação contínua)

- Treinado por **1.142.322 passos de ambiente** (5141 episódios), o mesmo orçamento dos 8000 episódios do DQN (1.142.006 passos). A comparação é feita por passos, e não por episódios, porque os episódios do DQN são curtos no começo e longos depois.
- Rede ator-crítico com 2 camadas de 64 (tanh), política gaussiana, `log_std` inicial = −1, normalização de recompensa; o PPO atualiza a cada rollout, não ao fim de cada episódio.
- O PPO usa o controle por torque e dois termos de recompensa a mais do que o DQN (bola presa e variação da ação, veja a tabela na seção 3). Eles foram necessários por falhas observadas só no PPO: primeiro o agente deixava a bola parada no canto entre o pé e a canela; depois a perna tremia.
- Durante o treino (política com ruído), a média móvel de 100 episódios passa de 20 embaixadinhas com ~123 mil passos (o DQN, com ~118 mil), passa de 50 com ~203 mil e, nos últimos 500 episódios, fica em ~94 por episódio, com 89% dos episódios chegando ao limite de 500 decisões e 7% terminando por bola presa. Os números de treino não são diretamente comparáveis: o DQN treina com 5% de ações aleatórias e o PPO sorteia ações de uma gaussiana.
- **Lição dos pilotos:** as métricas de treino não mostravam como a política determinística se comporta. Com `k_suavidade = 0`, os episódios de treino pareciam ótimos, mas na avaliação determinística todos terminavam com a bola presa. Por isso o `treino.py` agora avalia a política determinística durante o treino e guarda o melhor checkpoint (`modelo_melhor.pt`, `avaliacao.csv`).
- Avaliação final DQN × PPO (`python -m src.comparar`): mesmo protocolo para os dois (política gulosa/determinística, 3 sementes de ambiente, 600 episódios por agente). Cada agente foi testado com a regra de bola presa desligada (original) e ligada. Resultados em `resultados/comparacao/`:

  | Política | Embaixadinhas (regra original) | Embaixadinhas (com regra de bola presa) | Episódios completos (500 decisões) | Agitação da perna (rad/s por decisão) |
  |---|---|---|---|---|
  | Perna parada | 1,4 | – | 0% | 0,03 |
  | Aleatória | 0,2 | – | 0% | 6,02 |
  | DQN | 78,5 (mediana 80, máx. 92) | 78,3 | 93,8% | 4,42 |
  | **PPO** | **102,8** (mediana 108, máx. 114) | 102,8 | 95,0% | 3,29 |

  - A regra de bola presa quase não muda o jogo: o DQN perde 0,2 embaixadinha em média (1,2% dos episódios terminam presos) e o PPO não é afetado (0% presos).
  - Os dois chegam ao fim do episódio em ~94–95% dos casos, então a vantagem do PPO vem sobretudo do **ritmo** (mais toques por episódio de 500 decisões), e não de sobreviver mais.
  - Os dois falham de forma diferente: o DQN deixa a bola cair no meio do episódio (média de 323 decisões, ~48 embaixadinhas), enquanto todas as 30 falhas do PPO acontecem nas primeiras decisões (média de 23 decisões, no máximo 3 embaixadinhas): ele não alcança a bola do primeiro lançamento. Passando disso, nenhum episódio perdeu a bola.
  - A perna do PPO tem menos agitação (3,29) do que a do DQN (4,42).
  - Limitação: cada algoritmo foi treinado uma única vez (uma semente de treino). As 3 sementes da avaliação variam só o ambiente, não o treino.

## Entregas

- [x] Entrega 1: descrição do problema (apresentação em 29/10/2026)
- [x] Entrega 2: implementação do problema usando o algoritmo do Deep Q-Network
- [ ] Entrega 3: implementação com PPO (ação contínua) e comparação com o DQN (apresentação em 01/12/2026)
- As apresentações 2 e 3 foram mescladas em uma só, por orientação do professor.

## Créditos
 
- Imagem do esquema da perna (quadril, joelho e tornozelo) usada no slide 2 da primeira apresentação: H. Lv e B. Hao, *Convolutional neural network and long short-term memory hybrid model-based gait prediction for spacesuit intelligent assistive device in low gravity environment*.
 
