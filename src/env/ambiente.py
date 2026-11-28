"""
Ambiente 2D de embaixadinhas: uma perna com 3 articulações (quadril, joelho,
tornozelo) tenta manter uma bola no ar.

A interface segue o padrão do Gymnasium (reset / step), mas sem herdar dele:
    obs, info = env.reset()
    obs, recompensa, terminou, truncou, info = env.step(acao)

Modos de ação (configurados via parâmetro `acao_continua`):
- Discreto (`acao_continua=False`, padrão): um inteiro de 0 a 26. Cada articulação
  recebe um nível de velocidade-alvo em {-1, 0, +1} (girar para um lado, parar
  segurando a pose, girar para o outro), totalizando 3 x 3 x 3 = 27 combinações.
- Contínuo (`acao_continua=True`): um vetor `a = (τ₁, τ₂, τ₃) ∈ [-1, 1]³` com o torque
  normalizado de cada articulação. No Box2D o torque é aplicado pelo motor da junta: o
  sentido é o sinal de `a` e o torque máximo é `|a| * TORQUE_MAX`; a velocidade da junta
  fica limitada a `VEL_MAX` (acima disso o motor freia), o que faz o papel de amortecimento.
"""
import itertools
import math

import numpy as np
import pygame
from Box2D.b2 import (world, polygonShape, circleShape, contactListener)

# ----------------------------------------------------------------------------
# Constantes do mundo (unidades: metros, segundos, radianos)
# ----------------------------------------------------------------------------
LARGURA = 4.0            # largura da área (m)
ALTURA = 5.0             # altura da área (m)
GRAVIDADE = -9.8

DT = 1.0 / 60.0          # passo da física
FRAME_SKIP = 4           # a cada decisão do agente, a física avança 4 passos
MAX_PASSOS = 500         # limite do episódio (500 decisões = ~33 s simulados)

# Perna
QUADRIL = (1.3, 2.0)     # posição fixa do quadril
C_COXA, C_CANELA, C_PE = 0.8, 0.8, 0.5
ESPESSURA = 0.1
DENSIDADE_PERNA = 15.0           # perna pesada: não cede quando a bola bate nela
VEL_MAX = 3.0            # velocidade-alvo máxima do motor (rad/s)
TORQUE_MAX = 80.0        # torque máximo de cada motor (N.m); precisa vencer a gravidade da perna

# Pose inicial (ângulos absolutos de cada segmento) e limites de cada junta,
# medidos a partir da pose inicial (ângulo da junta = 0 na pose inicial)
ANG_COXA = -math.pi / 4          # coxa apontando para frente e para baixo
ANG_CANELA = -math.pi / 2        # canela vertical, para baixo
ANG_PE = 0.0                     # pé horizontal, apontando para frente
LIMITES = [(-0.9, 1.4),          # quadril
           (-1.5, 0.78),         # joelho
           (-0.6, 0.0)]          # tornozelo: só pode abrir (pé desce). Não pode fechar o "L"
                                 # do pé contra a canela, o que prenderia a bola

# Bola
RAIO_BOLA = 0.15
DENSIDADE_BOLA = 0.3             # bola leve
RESTITUICAO_BOLA = 0.5           # 0 = não quica, 1 = quica sem perder energia
ALTURA_INICIAL_BOLA = 4.0
SORTEIO_X = 0.2                  # a bola começa a até 0.2 m (para cada lado) do centro do pé

# Recompensas (mesmos valores do slide 3)
R_EMBAIXADINHA = 10.0
R_VIVO = 0.01
K_DISTANCIA = 0.1
K_ESFORCO = 0.001
R_QUEDA = -10.0
VY_MIN_TOQUE = 1.0               # a bola precisa sair do pé subindo a pelo menos 1 m/s
COOLDOWN_TOQUE = 12              # mínimo de passos de física (0,2 s) entre duas embaixadinhas

# Anti "bola presa": impede que o agente carregue/pare a bola contra a perna (por exemplo,
# no canto em "L" entre o pé e a canela) só para ganhar R_VIVO sem arriscar a queda.
# A bola conta como "encostada" quando a superfície dela está a menos de MARGEM_ENCOSTADA da
# perna (usa distância geométrica, então não é enganada pela bola "tremendo" no contato).
MARGEM_ENCOSTADA = 0.04          # m
GRACA_PRESO = 15                 # passos de física (0,25 s) encostada sem nenhuma penalidade
LIMITE_PRESO = 60                # passos de física (1 s) seguidos encostada: o episódio termina
K_PRESO = 0.5                    # penalidade por decisão depois da graça (acumula até o limite)
R_PRESO_FIM = R_QUEDA            # ao terminar por "bola presa" vale o mesmo que deixar cair

# Suavidade da ação contínua (contra a perna "tremendo")
K_SUAVIDADE = 0.0                # padrão desligado (o DQN não usa); o treino do PPO liga
                                 # penalidade = K_SUAVIDADE * ‖a_t - a_(t-1)‖²

# Filtros de colisão: quem pode bater em quem
CAT_CHAO, CAT_BOLA, CAT_PE, CAT_PERNA = 0x0001, 0x0002, 0x0004, 0x0008
BOLA_COLIDE_PERNA = True         # True: a bola também bate na coxa e na canela (perna sólida).
                                 # False: a bola só interage com o pé e o chão (e atravessa a perna).

# Todas as 27 combinações de níveis de motor: matriz de formato (27, 3)
ACOES = np.array(list(itertools.product([-1.0, 0.0, 1.0], repeat=3)),
                 dtype=np.float32)

# Desenho
ESCALA = 140                     # pixels por metro
MARGEM_PX = 30                   # faixa visível do chão, embaixo
LARG_PX, ALT_PX = int(LARGURA * ESCALA), int(ALTURA * ESCALA) + MARGEM_PX


class DetectorDeToque(contactListener):
    """Avisa quando o pé encosta na bola (o Box2D chama BeginContact sozinho)."""

    def __init__(self):
        contactListener.__init__(self)
        self.toque = False       # pé encostou na bola
        self.queda = False       # bola encostou no chão

    def BeginContact(self, contato):
        nomes = {contato.fixtureA.body.userData, contato.fixtureB.body.userData}
        if nomes == {"pe", "bola"}:
            self.toque = True
        elif nomes == {"chao", "bola"}:
            self.queda = True


class EmbaixadinhaEnv:
    n_acoes = len(ACOES)         # 3 (ação contínua) ou 27 (ação discreta)
    dim_obs = 10

    def __init__(self, render_mode=None, semente=None, acao_continua=False, anti_preso=False,
                 controle_continuo="torque", k_suavidade=K_SUAVIDADE, alfa_filtro=0.0):
        self.render_mode = render_mode       # None, "human" ou "rgb_array"
        self.rng = np.random.default_rng(semente)
        self.tela = None
        self.relogio = None
        self.mundo = None
        self.acao_continua = acao_continua
        self.n_acoes = 3 if acao_continua else 27
        # anti_preso=True: penaliza e depois encerra o episódio se a bola ficar encostada na
        # perna por muito tempo. Padrão False para manter o DQN já treinado reproduzível.
        self.anti_preso = anti_preso
        # Como a ação contínua a ∈ [-1, 1] vira comando do motor:
        #   "torque" (padrão, o das apresentações): sentido = sinal(a), torque máximo
        #                 |a| * TORQUE_MAX. Enquanto |ω| < VEL_MAX o torque aplicado é a *
        #                 TORQUE_MAX; acima disso o motor freia (limitador de velocidade).
        #   "velocidade" (alternativa/ablação): velocidade-alvo = a * VEL_MAX com torque máximo
        #                 fixo, a versão contínua das 27 ações do DQN (a = 0 segura a pose).
        self.controle_continuo = controle_continuo
        self.k_suavidade = k_suavidade       # penalidade por mudança brusca de ação
        self.alfa_filtro = alfa_filtro       # 0 = sem filtro; 0.5 = média móvel exponencial
        self.comando_anterior = np.zeros(3, dtype=np.float32)

    # ------------------------------------------------------------------
    # Construção do mundo
    # ------------------------------------------------------------------
    def _criar_segmento(self, ponto_inicial, angulo, comprimento, nome):
        """Cria um retângulo que começa em ponto_inicial e aponta na direção 'angulo'."""
        cx = ponto_inicial[0] + 0.5 * comprimento * math.cos(angulo)
        cy = ponto_inicial[1] + 0.5 * comprimento * math.sin(angulo)
        corpo = self.mundo.CreateDynamicBody(position=(cx, cy), angle=angulo,
                                             userData=nome)
        corpo.CreateFixture(
            shape=polygonShape(box=(comprimento / 2, ESPESSURA / 2)),
            density=DENSIDADE_PERNA, friction=0.8, restitution=0.0,
            groupIndex=-1,          # grupo negativo: partes da perna não colidem entre si
            categoryBits=CAT_PE if nome == "pe" else CAT_PERNA,
            maskBits=(CAT_BOLA | CAT_CHAO) if nome == "pe"
                     else CAT_CHAO | (CAT_BOLA if BOLA_COLIDE_PERNA else 0))
        fim = (ponto_inicial[0] + comprimento * math.cos(angulo),
               ponto_inicial[1] + comprimento * math.sin(angulo))
        return corpo, fim

    def _criar_junta(self, corpo_a, corpo_b, ancora, limites):
        return self.mundo.CreateRevoluteJoint(
            bodyA=corpo_a, bodyB=corpo_b, anchor=ancora,
            enableMotor=True, motorSpeed=0.0, maxMotorTorque=TORQUE_MAX,
            enableLimit=True, lowerAngle=limites[0], upperAngle=limites[1])

    def _construir_mundo(self):
        self.mundo = world(gravity=(0, GRAVIDADE), doSleep=False)
        self.detector = DetectorDeToque()
        self.mundo.contactListener = self.detector

        # Chão: retângulo estático cuja face de cima está em y = 0
        self.mundo.CreateStaticBody(
            position=(LARGURA / 2, -0.25), userData="chao",
            shapes=polygonShape(box=(LARGURA * 2, 0.25)))

        # Quadril: corpo estático (fixo no espaço), sem forma
        quadril = self.mundo.CreateStaticBody(position=QUADRIL)

        # Segmentos da perna, do quadril até o pé
        self.coxa, joelho_pos = self._criar_segmento(QUADRIL, ANG_COXA, C_COXA, "coxa")
        self.canela, tornozelo_pos = self._criar_segmento(joelho_pos, ANG_CANELA, C_CANELA, "canela")
        self.pe, _ = self._criar_segmento(tornozelo_pos, ANG_PE, C_PE, "pe")

        # Juntas com motor e limite de ângulo
        self.juntas = [
            self._criar_junta(quadril, self.coxa, QUADRIL, LIMITES[0]),
            self._criar_junta(self.coxa, self.canela, joelho_pos, LIMITES[1]),
            self._criar_junta(self.canela, self.pe, tornozelo_pos, LIMITES[2]),
        ]

        # Bola (criada em reset(), pois a posição inicial é sorteada)
        self.bola = None

    def _distancia_perna(self):
        """Menor distância (m) entre a superfície da bola e a perna (≈ 0 quando encostam).

        Mede em relação a cada segmento sólido (pé; e coxa/canela se a bola colide com elas),
        no referencial do próprio segmento, onde ele é uma caixa alinhada aos eixos.
        """
        segmentos = [(self.pe, C_PE / 2)]
        if BOLA_COLIDE_PERNA:
            segmentos += [(self.canela, C_CANELA / 2), (self.coxa, C_COXA / 2)]
        menor = float("inf")
        for corpo, meio_comp in segmentos:
            lx, ly = corpo.GetLocalPoint(self.bola.position)
            fora_x = max(abs(lx) - meio_comp, 0.0)
            fora_y = max(abs(ly) - ESPESSURA / 2, 0.0)
            menor = min(menor, math.hypot(fora_x, fora_y) - RAIO_BOLA)
        return menor

    # ------------------------------------------------------------------
    # Interface principal
    # ------------------------------------------------------------------
    def reset(self, semente=None):
        if semente is not None:
            self.rng = np.random.default_rng(semente)
        self._construir_mundo()

        # Posição inicial da bola: sorteada horizontalmente perto do pé
        x_pe = self.pe.worldCenter.x
        x_bola = x_pe + self.rng.uniform(-SORTEIO_X, SORTEIO_X)
        self.bola = self.mundo.CreateDynamicBody(
            position=(x_bola, ALTURA_INICIAL_BOLA), userData="bola")
        self.bola.CreateFixture(
            shape=circleShape(radius=RAIO_BOLA),
            density=DENSIDADE_BOLA, friction=0.3, restitution=RESTITUICAO_BOLA,
            categoryBits=CAT_BOLA,
            maskBits=CAT_CHAO | CAT_PE | (CAT_PERNA if BOLA_COLIDE_PERNA else 0))
        self.detector.toque = False
        self.detector.queda = False

        self.passo = 0
        self.passos_fisica = 0
        self.ultimo_toque = -COOLDOWN_TOQUE
        self.embaixadinhas = 0
        self.encostada = 0               # passos de física seguidos com a bola encostada na perna
        self.comando_anterior = np.zeros(3, dtype=np.float32)
        return self._observar(), {}

    def step(self, acao):
            # 1) Aplica a ação nos motores das 3 articulações
            if self.acao_continua:
                comando = np.clip(acao, -1.0, 1.0).astype(np.float32)
                if self.alfa_filtro > 0.0:       # filtro passa-baixa: corta oscilação rápida
                    comando = (self.alfa_filtro * self.comando_anterior
                               + (1.0 - self.alfa_filtro) * comando)
                for junta, u in zip(self.juntas, comando):
                    if self.controle_continuo == "torque":
                        junta.motorSpeed = float(np.sign(u)) * VEL_MAX
                        junta.maxMotorTorque = float(abs(u)) * TORQUE_MAX
                    else:
                        junta.motorSpeed = float(u) * VEL_MAX
                        junta.maxMotorTorque = TORQUE_MAX
            else:
                # Converte o número da ação (0 a 26) em um nível de motor por junta
                comando = ACOES[int(acao)]
                for junta, nivel in zip(self.juntas, comando):
                    junta.motorSpeed = float(nivel) * VEL_MAX
                    junta.maxMotorTorque = TORQUE_MAX

            # 2) Avança a física FRAME_SKIP vezes e conta embaixadinhas
            toques = 0
            tocou_chao = False
            preso = False
            for _ in range(FRAME_SKIP):
                self.mundo.Step(DT, 8, 3)
                self.passos_fisica += 1
                if self.detector.toque:
                    self.detector.toque = False
                    # Embaixadinha válida: a bola saiu do pé subindo com força, estava
                    # acima do pé e já passou o tempo mínimo desde a última contagem
                    # (evita contar vários toques seguidos quando a bola fica "tremendo")
                    if (self.bola.linearVelocity.y > VY_MIN_TOQUE
                            and self.bola.position.y > self.pe.worldCenter.y
                            and self.passos_fisica - self.ultimo_toque >= COOLDOWN_TOQUE):
                        toques += 1
                        self.ultimo_toque = self.passos_fisica
                if self.render_mode == "human":
                    self.render()
                if self.detector.queda:          # bola tocou o chão: encerra na hora,
                    tocou_chao = True            # sem esperar o fim da decisão
                    break
                if self.anti_preso:
                    if self._distancia_perna() <= MARGEM_ENCOSTADA:
                        self.encostada += 1
                    else:
                        self.encostada = 0
                    if self.encostada >= LIMITE_PRESO:   # bola carregada/parada na perna
                        preso = True
                        break
            self.embaixadinhas += toques
            self.passo += 1

            # 3) Fim de episódio
            bx, by = self.bola.position
            caiu = tocou_chao or by <= RAIO_BOLA + 0.02
            saiu = bx < 0 or bx > LARGURA or by > ALTURA + 3.0
            terminou = bool(caiu or saiu or preso)
            truncou = self.passo >= MAX_PASSOS and not terminou

            # 4) Recompensa
            dx = abs(bx - self.pe.worldCenter.x)
            # Com anti_preso, "bola no ar" só rende R_VIVO se ela NÃO está encostada na perna
            vivo = R_VIVO if not (self.anti_preso and self.encostada > 0) else 0.0
            recompensa = (vivo
                        - K_DISTANCIA * dx
                        - K_ESFORCO * float(np.sum(comando ** 2))
                        + R_EMBAIXADINHA * toques)
            if self.k_suavidade > 0.0:
                variacao = comando - self.comando_anterior
                recompensa -= self.k_suavidade * float(np.sum(variacao ** 2))
            self.comando_anterior = np.asarray(comando, dtype=np.float32)
            if self.anti_preso and self.encostada > GRACA_PRESO:
                recompensa -= K_PRESO                    # cada decisão "carregando" custa
            if preso:
                recompensa += R_PRESO_FIM
            elif terminou:
                recompensa += R_QUEDA

            info = {"embaixadinhas": self.embaixadinhas, "preso": preso}
            return self._observar(), recompensa, terminou, truncou, info

    # ------------------------------------------------------------------
    # Observação (estado markoviano), já normalizada para ~[-1, 1]
    # ------------------------------------------------------------------
    def _observar(self):
        bx, by = self.bola.position
        bvx, bvy = self.bola.linearVelocity
        angulos = [j.angle for j in self.juntas]
        velocidades = [j.speed for j in self.juntas]
        obs = [
            (bx - LARGURA / 2) / (LARGURA / 2),
            (by - ALTURA / 2) / (ALTURA / 2),
            bvx / 5.0,
            bvy / 10.0,
            *[a / 1.5 for a in angulos],
            *[w / VEL_MAX for w in velocidades],
        ]
        return np.clip(np.array(obs, dtype=np.float32), -5.0, 5.0)

    # ------------------------------------------------------------------
    # Desenho (pygame)
    # ------------------------------------------------------------------
    def _px(self, ponto):
        return int(ponto[0] * ESCALA), int(ALT_PX - MARGEM_PX - ponto[1] * ESCALA)

    def render(self):
        if self.tela is None:
            pygame.init()
            if self.render_mode == "human":
                self.tela = pygame.display.set_mode((LARG_PX, ALT_PX))
                self.relogio = pygame.time.Clock()
            else:
                self.tela = pygame.Surface((LARG_PX, ALT_PX))
            self.fonte = pygame.font.SysFont(None, 28)

        self.tela.fill((15, 27, 45))
        pygame.draw.rect(self.tela, (60, 80, 110), (0, ALT_PX - MARGEM_PX, LARG_PX, MARGEM_PX))
        for corpo, cor in ((self.coxa, (244, 241, 234)),
                           (self.canela, (244, 241, 234)),
                           (self.pe, (255, 138, 61))):
            vertices = [self._px(corpo.transform * v) for v in corpo.fixtures[0].shape.vertices]
            pygame.draw.polygon(self.tela, cor, vertices)
        for junta in self.juntas:
            pygame.draw.circle(self.tela, (74, 168, 255), self._px(junta.anchorA), 7)
        pygame.draw.circle(self.tela, (255, 255, 255), self._px(self.bola.position),
                           int(RAIO_BOLA * ESCALA))
        texto = self.fonte.render(f"Embaixadinhas: {self.embaixadinhas}", True, (244, 241, 234))
        self.tela.blit(texto, (12, 10))

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self.relogio.tick(60)
        else:
            return np.transpose(pygame.surfarray.array3d(self.tela), (1, 0, 2))

    def close(self):
        if self.tela is not None:
            pygame.quit()
            self.tela = None
