# Maze Solver — Reporte de Avance

## Qué se hizo

Se implementaron los dos primeros módulos del proyecto: el **parser de laberintos** y el **ambiente de simulación**.

---

## 1. Parser (`maze_parser.py`)

**Problema:** los laberintos se entregaron como imágenes PNG (1026×1026 px), no como arrays. Había que convertirlos a una estructura usable por el agente.

**Cómo se resolvió:**

- Cada celda ocupa un bloque de 16×16 px. El laberinto es passage-based: todas las celdas son transitables y las paredes son los bordes de 2 px entre celdas.
- **Detección de paredes:** se samplea el pixel de borde entre dos celdas adyacentes. Negro (0) = pared, blanco (255) = pasaje abierto. Esto produce dos arrays booleanos `can_go_right` y `can_go_down` de 64×64.
- **Detección de hazards:** se analiza el color promedio de los píxeles no-blancos dentro de cada celda en `MAZE_1.png` y se clasifica por perfil de color:
  - Naranja (🔥) → Death pit
  - Círculo verde sólido (🟢) → Start
  - Círculo dorado (🟡) → Goal
  - Círculo púrpura (🟣) → Teleport pad
  - Calavera / snowflake → Confusion trap

**Resultado:** BFS desde el start llega a las 4096 celdas en ambos laberintos, confirmando que la extracción de paredes es correcta.

| Laberinto | Start | Goal | Pits | Teleports | Confusions |
|---|---|---|---|---|---|
| Alpha (training) | (31, 35) | (55, 59) | 35 | 2 | 4 |
| Gamma (testing)  | (52, 16) | (47, 35) | 53 | 8 | 4 |

---

## 2. Ambiente (`environment.py`)

**Problema:** necesitamos simular localmente el API del profesor para poder entrenar y testear el agente.

**Cómo se resolvió:**

Se implementaron exactamente las clases del spec (`Action`, `TurnResult`, `MazeEnvironment`, `Agent`) con todos los mecánismos:

- **Paredes:** si el movimiento está bloqueado, incrementa `wall_hits` y la posición no cambia.
- **Death pits:** el agente muere, `is_dead=True`, respawnea en start el siguiente turno. El episodio continúa.
- **Teleports:** al pisar un pad, la posición salta al destino configurado. Soporta cadenas (multi-hop).
- **Confusion:** al pisar una celda de confusión, UP↔DOWN y LEFT↔RIGHT se invierten durante el resto del turno actual y el turno siguiente completo.

**Verificado:**
- Muerte en pit → respawn correcto ✓
- `wall_hits` al chocar con pared ✓
- BFS encuentra camino de 267 pasos de start a goal ✓

---

## Pendiente

- Destinos de los teleports son placeholders — el agente los descubrirá por exploración.
- Implementar el agente naive (BFS explorer).
- Implementar el agente inteligente (A\* + RL).

---
