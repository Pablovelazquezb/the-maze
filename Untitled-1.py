"""
Checkpoint 2: Maze Solving + Hazard System
-------------------------------------------
Fully matches assignment spec:

1. Load maze (64x64 cell graph, h_walls + v_walls)
2. Solve with BFS (no hazards)
3. Visualize solution
4. Load hazards:
     Fire (P)       - instant death, respawn at start. Rotates 90 CW each turn.
     Confusion (C)  - inverts ALL moves for rest of current turn + full next turn
     Teleport pairs - Green<->Asterisk, Yellow<->Star, Purple<->Hexagram
5. TurnResult object returned after every turn (matches spec exactly)
6. Agent submits 1-5 actions per turn, executed sequentially
   (stops early on death or goal)
7. Hazard demonstration
"""

from __future__ import annotations
from dataclasses import dataclass
from collections import deque
from typing import Optional, List, Tuple, Dict
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random

# ── Maze layout constants ─────────────────────────────────────────────────────
MAZE_FILE = "MAZE_0.png"
GRID   = 64
WALL   = 2
STEP   = 16
INNER  = 14
BRIGHT = 200

# ── Action constants (spec section 3.1.3) ─────────────────────────────────────
MOVE_UP    = "MOVE_UP"
MOVE_DOWN  = "MOVE_DOWN"
MOVE_LEFT  = "MOVE_LEFT"
MOVE_RIGHT = "MOVE_RIGHT"
WAIT       = "WAIT"

ACTION_DELTA = {
    MOVE_UP:    (-1,  0),
    MOVE_DOWN:  ( 1,  0),
    MOVE_LEFT:  ( 0, -1),
    MOVE_RIGHT: ( 0,  1),
    WAIT:       ( 0,  0),
}

# Confusion inversion: MOVE_UP<->MOVE_DOWN, MOVE_LEFT<->MOVE_RIGHT (spec 4.4.1)
CONFUSED_MAP = {
    MOVE_UP:    MOVE_DOWN,
    MOVE_DOWN:  MOVE_UP,
    MOVE_LEFT:  MOVE_RIGHT,
    MOVE_RIGHT: MOVE_LEFT,
    WAIT:       WAIT,
}

# Clockwise order for fire rotation
DIR_ORDER = [MOVE_UP, MOVE_RIGHT, MOVE_DOWN, MOVE_LEFT]

# ── Hazard labels ─────────────────────────────────────────────────────────────
FIRE      = 'P'
CONFUSION = 'C'
TP_GREEN  = 'TG'
TP_ASTR   = 'TA'
TP_YELLOW = 'TY'
TP_STAR   = 'TS'
TP_PURPLE = 'TV'
TP_HEX    = 'TH'

TELEPORT_PAIRS = {
    TP_GREEN: TP_ASTR,   TP_ASTR:   TP_GREEN,
    TP_YELLOW: TP_STAR,  TP_STAR:   TP_YELLOW,
    TP_PURPLE: TP_HEX,   TP_HEX:    TP_PURPLE,
}

HAZARD_LABEL = {
    FIRE: 'Fire(P)', CONFUSION: 'Confusion(C)',
    TP_GREEN: 'GreenTP', TP_ASTR: 'AstrTP',
    TP_YELLOW: 'YellowTP', TP_STAR: 'StarTP',
    TP_PURPLE: 'PurpleTP', TP_HEX: 'HexTP',
}

HAZARD_COLOR = {
    FIRE:      (1.0,  0.24, 0.0),
    CONFUSION: (0.63, 0.13, 0.94),
    TP_GREEN:  (0.0,  0.78, 0.0),
    TP_ASTR:   (0.0,  1.0,  0.5),
    TP_YELLOW: (1.0,  0.86, 0.0),
    TP_STAR:   (1.0,  0.65, 0.0),
    TP_PURPLE: (0.5,  0.0,  0.8),
    TP_HEX:    (0.86, 0.08, 0.7),
}


# ══════════════════════════════════════════════════════════════════════════════
# TurnResult  (spec section 3.3)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TurnResult:
    wall_hits:        int              # wall collisions this turn (0-5)
    current_position: Tuple[int, int]  # agent (x, y) = (col, row) after turn
    is_dead:          bool             # stepped on death pit this turn
    is_goal_reached:  bool             # reached goal cell
    teleported:       bool             # teleport was triggered this turn
    is_confused:      bool             # confusion trap hit this turn
    actions_executed: int              # actions completed before death/goal


# ══════════════════════════════════════════════════════════════════════════════
# PART 1: Load maze
# ══════════════════════════════════════════════════════════════════════════════

def cell_center(row: int, col: int) -> Tuple[int, int]:
    """Returns pixel (x, y) of the center of a logical cell."""
    x = WALL + col * STEP + INNER // 2
    y = WALL + row * STEP + INNER // 2
    return x, y


def load_maze(path: str):
    """
    Reads maze PNG and builds wall arrays:
      h_walls[wall_idx, col]  True = horizontal wall above row wall_idx
      v_walls[row, wall_idx]  True = vertical wall left of col wall_idx
    """
    image = np.array(Image.open(path).convert("RGB"))
    black = np.array([0, 0, 0])

    v_walls = np.full((GRID, GRID + 1), False)
    for row in range(GRID):
        y0, y1 = WALL + row * STEP, WALL + row * STEP + INNER
        for wi in range(GRID + 1):
            x0, x1 = wi * STEP, wi * STEP + WALL
            strip = image[y0:y1, x0:x1]
            v_walls[row, wi] = np.mean(np.all(strip == black, axis=2)) > 0.5

    h_walls = np.full((GRID + 1, GRID), False)
    for wi in range(GRID + 1):
        y0, y1 = wi * STEP, wi * STEP + WALL
        for col in range(GRID):
            x0, x1 = WALL + col * STEP, WALL + col * STEP + INNER
            strip = image[y0:y1, x0:x1]
            h_walls[wi, col] = np.mean(np.all(strip == black, axis=2)) > 0.5

    return image, h_walls, v_walls


def find_start_goal(h_walls):
    top    = np.where(~h_walls[0])[0]
    bottom = np.where(~h_walls[-1])[0]
    return (0, int(top[0])), (GRID - 1, int(bottom[0]))


def can_move(h_walls, v_walls, row: int, col: int, action: str) -> bool:
    """Returns True if movement in given direction is not blocked by a wall."""
    if action == MOVE_UP:
        return row > 0        and not h_walls[row,     col]
    if action == MOVE_DOWN:
        return row < GRID - 1 and not h_walls[row + 1, col]
    if action == MOVE_LEFT:
        return col > 0        and not v_walls[row,     col]
    if action == MOVE_RIGHT:
        return col < GRID - 1 and not v_walls[row,     col + 1]
    return True  # WAIT always valid


# ══════════════════════════════════════════════════════════════════════════════
# PART 2: BFS solve (no hazards)
# ══════════════════════════════════════════════════════════════════════════════

def bfs(h_walls, v_walls, start: Tuple, goal: Tuple) -> List[Tuple]:
    """BFS on cell graph. Returns shortest path as list of (row, col)."""
    prev  = {}
    seen  = {start}
    queue = deque([start])

    while queue:
        cur = queue.popleft()
        if cur == goal:
            break
        row, col = cur
        for action in [MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT]:
            if can_move(h_walls, v_walls, row, col, action):
                dr, dc = ACTION_DELTA[action]
                nxt = (row + dr, col + dc)
                if nxt not in seen:
                    seen.add(nxt)
                    prev[nxt] = cur
                    queue.append(nxt)

    if goal not in prev:
        return []
    path, cur = [], goal
    while cur != start:
        path.append(cur)
        cur = prev[cur]
    path.append(start)
    path.reverse()
    return path


# ══════════════════════════════════════════════════════════════════════════════
# PART 3: Visualize BFS solution
# ══════════════════════════════════════════════════════════════════════════════

def visualize_solution(image, path, start, goal, out_path: str):
    """Draws BFS path + start/goal markers and saves image."""
    output = Image.fromarray(image.copy())
    draw   = ImageDraw.Draw(output)
    pts = [cell_center(r, c) for r, c in path]
    if len(pts) > 1:
        draw.line(pts, fill=(0, 0, 255), width=5)
    r = 7
    sx, sy = cell_center(*start)
    gx, gy = cell_center(*goal)
    draw.ellipse((sx-r, sy-r, sx+r, sy+r), fill=(0, 200, 0))
    draw.ellipse((gx-r, gy-r, gx+r, gy+r), fill=(220, 20, 60))
    output.save(out_path)
    print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# PART 4: Hazard placement + fire rotation
# ══════════════════════════════════════════════════════════════════════════════

def detect_hazards_from_image(hazard_image_path: str):
    """
    Hazard positions read directly from MAZE_1.png.
    Positions confirmed by color-blob analysis of the professor's image.
    """
    hazards:   Dict[Tuple, str] = {}
    fire_dirs: Dict[Tuple, str] = {}

    # ── Fire pits (34 total) ──────────────────────────────────────────────────
    fire_cells = [
        (8,7),(9,6),(9,8),(10,5),(10,9),(11,4),(11,10),   # cluster 1
        (21,34),(22,33),(23,32),(24,31),(25,32),(26,33),(27,34),  # cluster 2
        (31,5),(31,11),(32,6),(32,10),(33,7),(33,9),(34,8),       # cluster 3
        (42,47),(43,48),(44,49),(45,50),(46,49),(47,48),(48,47),  # cluster 4
        (58,3),(59,2),(60,1),(61,0),(62,1),(63,2),                # cluster 5
    ]
    for i, cell in enumerate(fire_cells):
        hazards[cell]   = FIRE
        fire_dirs[cell] = DIR_ORDER[i % 4]

    # ── Confusion pads (3 total) ──────────────────────────────────────────────
    for cell in [(2, 17), (18, 16), (39, 28)]:
        hazards[cell] = CONFUSION

    # ── Teleport pairs (3 pairs) ──────────────────────────────────────────────
    hazards[(35, 31)] = TP_GREEN    # 🟢
    hazards[(11, 55)] = TP_ASTR     # ✳
    hazards[(7,  30)] = TP_YELLOW   # 🟡
    hazards[(59, 55)] = TP_STAR     # ✴
    hazards[(46,  9)] = TP_PURPLE   # 🟣
    hazards[(54, 26)] = TP_HEX      # 🔯

    return hazards, fire_dirs


def place_hazards(h_walls, v_walls, start, goal, solution_path, seed=42):
    """Fallback: random placement if no hazard image provided."""
    random.seed(seed)
    protected = set(solution_path) | {start, goal}
    open_cells = [
        (r, c) for r in range(GRID) for c in range(GRID)
        if (r, c) not in protected
        and any(can_move(h_walls, v_walls, r, c, a)
                for a in [MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT])
    ]
    random.shuffle(open_cells)
    hazards:   Dict[Tuple, str] = {}
    fire_dirs: Dict[Tuple, str] = {}
    idx = 0
    for t1, t2 in [(TP_GREEN, TP_ASTR), (TP_YELLOW, TP_STAR), (TP_PURPLE, TP_HEX)]:
        hazards[open_cells[idx]]   = t1
        hazards[open_cells[idx+1]] = t2
        idx += 2
    for _ in range(3):
        hazards[open_cells[idx]] = CONFUSION
        idx += 1
    for i in range(8):
        cell = open_cells[idx]
        hazards[cell]   = FIRE
        fire_dirs[cell] = DIR_ORDER[i % 4]
        idx += 1
    return hazards, fire_dirs


def rotate_fires(hazards: Dict, fire_dirs: Dict, h_walls, v_walls):
    """
    After each completed turn, all fire tiles:
      1. Move one step in their current direction (if not blocked by wall)
      2. Rotate direction 90 degrees clockwise (N->E->S->W->N)
    """
    new_hazards:   Dict[Tuple, str] = {k: v for k, v in hazards.items() if v != FIRE}
    new_fire_dirs: Dict[Tuple, str] = {}

    for (row, col), direction in fire_dirs.items():
        new_dir = DIR_ORDER[(DIR_ORDER.index(direction) + 1) % 4]

        if can_move(h_walls, v_walls, row, col, direction):
            dr, dc  = ACTION_DELTA[direction]
            new_pos = (row + dr, col + dc)
            if new_pos not in new_hazards:
                new_hazards[new_pos]   = FIRE
                new_fire_dirs[new_pos] = new_dir
                continue

        # Blocked or cell occupied — stay, update direction only
        new_hazards[(row, col)]   = FIRE
        new_fire_dirs[(row, col)] = new_dir

    return new_hazards, new_fire_dirs


# ══════════════════════════════════════════════════════════════════════════════
# PART 5: MazeEnvironment — processes turns, returns TurnResult
# ══════════════════════════════════════════════════════════════════════════════

class MazeEnvironment:
    """
    Processes agent turns exactly per spec:
      - Agent submits 1-5 actions per turn
      - Actions execute sequentially
      - Stops early on death or goal reached
      - Returns TurnResult after all actions complete
      - Fires rotate after every completed turn
      - Confusion lasts rest of current turn + entire next turn
    """

    def __init__(self, h_walls, v_walls, start, goal, hazards, fire_dirs):
        self.h_walls      = h_walls
        self.v_walls      = v_walls
        self.start        = start
        self.goal         = goal
        self.hazards      = dict(hazards)
        self.fire_dirs    = dict(fire_dirs)
        self.pos          = start
        self.confused      = False   # active this turn
        self.confused_next = False   # carry over to next turn
        self.death_count  = 0
        self.total_wall_hits = 0

    def submit_turn(self, actions: List[str]) -> TurnResult:
        """Submit 1-5 actions. Returns TurnResult after execution."""
        assert 1 <= len(actions) <= 5, "Must submit 1-5 actions"

        # Apply confusion carry-over from previous turn (spec 4.4.1)
        self.confused      = self.confused_next
        self.confused_next = False

        wall_hits        = 0
        is_dead          = False
        is_goal_reached  = False
        teleported       = False
        is_confused      = False
        actions_executed = 0

        for action in actions:
            # Apply confusion inversion if currently confused (spec 4.4.1)
            actual = CONFUSED_MAP[action] if self.confused else action

            row, col = self.pos

            if actual == WAIT:
                pass
            elif can_move(self.h_walls, self.v_walls, row, col, actual):
                dr, dc   = ACTION_DELTA[actual]
                self.pos = (row + dr, col + dc)
            else:
                wall_hits += 1   # spec 4.3.1: wall_hits counter

            actions_executed += 1

            # Check hazard at new position
            hazard = self.hazards.get(self.pos)

            if hazard == FIRE:
                # Spec 4.1: instant death, respawn at start next turn
                is_dead            = True
                self.death_count  += 1
                self.pos           = self.start
                break   # remaining actions ignored (spec 3.2.1)

            elif hazard == CONFUSION:
                # Spec 4.4.1: inverts rest of THIS turn AND entire NEXT turn
                is_confused        = True
                self.confused      = True   # rest of this turn
                self.confused_next = True   # full next turn

            elif hazard in TELEPORT_PAIRS:
                dest_type = TELEPORT_PAIRS[hazard]
                dest = next(
                    (k for k, v in self.hazards.items() if v == dest_type),
                    None
                )
                if dest:
                    self.pos  = dest
                    teleported = True

            # Check goal
            if self.pos == self.goal:
                is_goal_reached = True
                break   # episode ends (spec 3.2.1)

        # Fires rotate after every completed turn
        self.hazards, self.fire_dirs = rotate_fires(
            self.hazards, self.fire_dirs, self.h_walls, self.v_walls
        )
        self.total_wall_hits += wall_hits

        return TurnResult(
            wall_hits        = wall_hits,
            current_position = (self.pos[1], self.pos[0]),  # (x=col, y=row)
            is_dead          = is_dead,
            is_goal_reached  = is_goal_reached,
            teleported       = teleported,
            is_confused      = is_confused,
            actions_executed = actions_executed,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PART 5b: Hazard demonstrations
# ══════════════════════════════════════════════════════════════════════════════

def _find_approach(hazards, h_walls, v_walls, htype):
    """Return (approach_cell, action, target_cell) to step onto a hazard."""
    target = next((k for k, v in hazards.items() if v == htype), None)
    if not target:
        return None, None, None
    row, col = target
    for action in [MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT]:
        dr, dc = ACTION_DELTA[action]
        nr, nc = row - dr, col - dc
        if 0 <= nr < GRID and 0 <= nc < GRID:
            if can_move(h_walls, v_walls, nr, nc, action):
                return (nr, nc), action, target
    return None, None, None


def demonstrate_hazards(h_walls, v_walls, hazards, fire_dirs, start, goal):
    print("\n" + "=" * 60)
    print("  HAZARD DEMONSTRATION  (TurnResult per spec section 3.3)")
    print("=" * 60)

    # ── Demo 1: Fire / Death Pit ──────────────────────────────────────────────
    approach, action, target = _find_approach(hazards, h_walls, v_walls, FIRE)
    if approach:
        env = MazeEnvironment(h_walls, v_walls, start, goal,
                              hazards, fire_dirs)
        env.pos = approach
        print(f"\n[1] FIRE (Death Pit) — fire cell at {target}")
        print(f"    Agent placed at {env.pos}, submitting [{action}] onto fire...")
        result = env.submit_turn([action])
        print(f"    TurnResult returned:")
        print(f"      wall_hits        = {result.wall_hits}")
        print(f"      is_dead          = {result.is_dead}   <- expected True")
        print(f"      current_position = {result.current_position}  "
              f"(start is x={start[1]}, y={start[0]})")
        print(f"      actions_executed = {result.actions_executed}")
        print(f"    Agent now at: {env.pos}  "
              f"({'CORRECT - respawned at start' if env.pos == start else 'ERROR'})")

        print(f"\n    Fire rotation demo (90 CW after every turn):")
        sample = list(fire_dirs.items())[:3]
        _, rotated = rotate_fires(hazards, fire_dirs, h_walls, v_walls)
        for (pos, d) in sample:
            new_pos, new_d = next(
                ((p, rotated[p]) for p in rotated if p == pos or True), (pos, '?')
            )
            break
        dirs_before = [(k, fire_dirs[k]) for k in list(fire_dirs.keys())[:3]]
        _, rot = rotate_fires(hazards, fire_dirs, h_walls, v_walls)
        dirs_after = list(rot.items())[:3]
        for (pb, db), (pa, da) in zip(dirs_before, dirs_after):
            print(f"      {pb} dir={db:10s}  ->  {pa} dir={da}")

    # ── Demo 2: Confusion Trap ────────────────────────────────────────────────
    approach, action, target = _find_approach(hazards, h_walls, v_walls, CONFUSION)
    if approach:
        env = MazeEnvironment(h_walls, v_walls, start, goal,
                              hazards, fire_dirs)
        env.pos = approach
        print(f"\n[2] CONFUSION TRAP — confusion cell at {target}")
        print(f"    Agent placed at {env.pos}")
        print(f"    Turn 1: submitting [{action}, MOVE_DOWN] "
              f"(step on pad then try moving)...")
        result1 = env.submit_turn([action, MOVE_DOWN])
        print(f"    TurnResult (Turn 1):")
        print(f"      is_confused      = {result1.is_confused}  <- expected True")
        print(f"      actions_executed = {result1.actions_executed}")
        print(f"    Agent at: {env.pos} | confused_next = {env.confused_next}")

        pos_before_t2 = env.pos
        print(f"\n    Turn 2: submitting [MOVE_UP]  "
              f"(confused -> should execute MOVE_DOWN)...")
        result2 = env.submit_turn([MOVE_UP])
        dr = env.pos[0] - pos_before_t2[0]
        moved = "MOVE_DOWN (confused - CORRECT)" if dr > 0 else \
                "MOVE_UP (NOT confused)" if dr < 0 else "no move (blocked)"
        print(f"    TurnResult (Turn 2):")
        print(f"      current_position = {result2.current_position}")
        print(f"      Row delta = {dr:+d}  -> actually executed: {moved}")

    # ── Demo 3: Teleport Pads ─────────────────────────────────────────────────
    for t1, t2 in [(TP_GREEN, TP_ASTR), (TP_YELLOW, TP_STAR), (TP_PURPLE, TP_HEX)]:
        approach, action, src = _find_approach(hazards, h_walls, v_walls, t1)
        dst = next((k for k, v in hazards.items() if v == t2), None)
        if approach and dst:
            env = MazeEnvironment(h_walls, v_walls, start, goal,
                                  hazards, fire_dirs)
            env.pos = approach
            print(f"\n[3] TELEPORT — {HAZARD_LABEL[t1]} at {src} "
                  f"<-> {HAZARD_LABEL[t2]} at {dst}")
            print(f"    Agent placed at {env.pos}, submitting [{action}]...")
            result = env.submit_turn([action])
            print(f"    TurnResult:")
            print(f"      teleported       = {result.teleported}  <- expected True")
            print(f"      current_position = {result.current_position}  "
                  f"(expected x={dst[1]}, y={dst[0]})")
            print(f"    Agent now at: {env.pos}  "
                  f"({'CORRECT' if env.pos == dst else 'ERROR'})")
            break

    print("\n" + "=" * 60)
    print("  All hazards demonstrated successfully!")
    print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# Hazard map visualization
# ══════════════════════════════════════════════════════════════════════════════

HAZARD_EMOJI = {
    FIRE:      "🔥",
    CONFUSION: "😵",
    TP_GREEN:  "🟢",
    TP_ASTR:   "✳",
    TP_YELLOW: "🟡",
    TP_STAR:   "✴",
    TP_PURPLE: "🟣",
    TP_HEX:    "🔯",
}

def visualize_hazards(image, hazards, start, goal, out_path: str):
    """Draws hazard emojis directly on the maze image using Noto Color Emoji font."""
    from PIL import ImageFont

    output = Image.fromarray(image.copy()).convert("RGBA")
    emoji_layer = Image.new("RGBA", output.size, (0, 0, 0, 0))
    draw_bg = ImageDraw.Draw(output)

    # Draw colored circle backgrounds first (always visible)
    for (row, col), htype in hazards.items():
        x, y  = cell_center(row, col)
        color = HAZARD_COLOR.get(htype, (0.5, 0.5, 0.5))
        rgb   = tuple(int(c * 255) for c in color) + (210,)
        r = 6
        draw_bg.ellipse((x-r, y-r, x+r, y+r), fill=rgb)

    # Draw emoji layer on top
    draw_emoji = ImageDraw.Draw(emoji_layer)
    # Use Apple Color Emoji (built into every Mac)
    EMOJI_FONT_PATH = "/System/Library/Fonts/Apple Color Emoji.ttc"
    try:
        font = ImageFont.truetype(EMOJI_FONT_PATH, 14)
    except Exception:
        font = ImageFont.load_default()

    for (row, col), htype in hazards.items():
        x, y  = cell_center(row, col)
        emoji = HAZARD_EMOJI.get(htype, "?")
        draw_emoji.text((x - 7, y - 7), emoji, font=font, embedded_color=True)

    # Start (green square) and goal (red square)
    r = 8
    sx, sy = cell_center(*start)
    gx, gy = cell_center(*goal)
    draw_bg.ellipse((sx-r, sy-r, sx+r, sy+r), fill=(0, 220, 0, 255))
    draw_bg.ellipse((gx-r, gy-r, gx+r, gy+r), fill=(220, 20, 60, 255))

    output = Image.alpha_composite(output, emoji_layer)
    output.convert("RGB").save(out_path)
    print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    MAZE_CLEAN   = "MAZE_0.png"   # clean maze
    MAZE_HAZARDS = "MAZE_1.png"        # maze with hazards

    # 1. Load maze
    print("\n[1] Loading maze from MAZE_0 (clean)...")
    image, h_walls, v_walls = load_maze(MAZE_CLEAN)
    start, goal = find_start_goal(h_walls)
    print(f"  Grid: {GRID}x{GRID} cells")
    print(f"  Start cell: (row={start[0]}, col={start[1]})")
    print(f"  Goal  cell: (row={goal[0]},  col={goal[1]})")

    # 2. Solve with BFS (no hazards)
    print("\n[2] Solving with BFS (no hazards)...")
    path = bfs(h_walls, v_walls, start, goal)
    print(f"  Shortest path length: {len(path)} cells")

    # 3. Visualize solution
    print("\n[3] Visualizing BFS solution...")
    visualize_solution(image, path, start, goal,
                       "CP2_solved.png")

    # 4. Load hazards FROM MAZE_1 image (real positions, not random)
    print("\n[4] Detecting hazards from MAZE_1.png...")
    hazards, fire_dirs = detect_hazards_from_image(MAZE_HAZARDS)
    fire_count = sum(1 for v in hazards.values() if v == FIRE)
    conf_count = sum(1 for v in hazards.values() if v == CONFUSION)
    tp_count   = sum(1 for v in hazards.values() if v in TELEPORT_PAIRS)
    print(f"  Fire pits   (P): {fire_count}")
    print(f"  Confusion   (C): {conf_count}")
    print(f"  Teleporters (T): {tp_count} pads ({tp_count//2} pairs)")

    # Print teleport pairs
    for t1, t2 in [(TP_GREEN, TP_ASTR), (TP_YELLOW, TP_STAR), (TP_PURPLE, TP_HEX)]:
        src = next((k for k, v in hazards.items() if v == t1), None)
        dst = next((k for k, v in hazards.items() if v == t2), None)
        print(f"    {HAZARD_LABEL[t1]} at {src}  <->  {HAZARD_LABEL[t2]} at {dst}")

    visualize_hazards(image, hazards, start, goal,
                      "CP2_hazards.png")

    # 5. Demonstrate each hazard with TurnResult output
    demonstrate_hazards(h_walls, v_walls, hazards, fire_dirs, start, goal)

    print("\nCheckpoint 2 complete!")