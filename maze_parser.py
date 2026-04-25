"""
Maze PNG Parser
Converts MAZE_0.png (structure) + MAZE_1.png (hazards) into numpy arrays.

Image layout: 1026x1026 pixels for a 64x64 cell grid.
  - 2px outer border + 64 * (14px cell interior + 2px wall separator) = 1026px
  - Cell (col, row) interior occupies pixels [col*16+2 .. col*16+15, row*16+2 .. row*16+15]
  - Wall between cells is detected at the shared boundary pixel

Start / Goal are the openings in the outer border (white gap in the 2px outer wall):
  - Top border opening  → Start (agent spawn)
  - Bottom border opening → Goal (exit)

Cell type encoding (matches environment spec):
  0 = EMPTY           - navigable
  2 = START           - agent spawn   (border opening, top)
  3 = GOAL            - target exit   (border opening, bottom)
  4 = PIT             - death trap    (fire emoji 🔥)
  5 = TELEPORT        - teleport pad  (purple circle 🟣)
  6 = CONFUSION       - inverts controls (skull / snowflake emoji)
  7 = TELEPORT_DEST   - destination of a teleport pad (green/gold circle markers)
"""

from __future__ import annotations
import numpy as np
from pathlib import Path
from PIL import Image


# ─── constants ────────────────────────────────────────────────────────────────

GRID_SIZE = 64
CELL_PX   = 16          # pixels per cell (including 2px right/bottom border)
BORDER    = 2           # outer border thickness

CELL_EMPTY     = 0
CELL_START     = 2
CELL_GOAL      = 3
CELL_PIT       = 4
CELL_TELEPORT  = 5
CELL_CONFUSION = 6


# ─── wall connectivity ────────────────────────────────────────────────────────

def extract_walls(img_gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (can_go_right, can_go_down), both shape (64, 64), dtype bool.
    can_go_right[row, col] = True  →  no wall between (col,row) and (col+1,row)
    can_go_down [row, col] = True  →  no wall between (col,row) and (col,row+1)

    Wall detection: sample the boundary pixel between two adjacent cells.
    The pixel at x = col*16 + 16  (the 2px separator's first pixel)
    is black (0) if blocked, white (255) if open passage.
    """
    N = GRID_SIZE
    can_go_right = np.zeros((N, N), dtype=bool)
    can_go_down  = np.zeros((N, N), dtype=bool)

    for row in range(N):
        cy = row * CELL_PX + BORDER + (CELL_PX - BORDER) // 2   # vertical center of cell

        for col in range(N):
            cx = col * CELL_PX + BORDER + (CELL_PX - BORDER) // 2  # horizontal center

            if col < N - 1:
                wall_x = col * CELL_PX + CELL_PX   # first boundary pixel to the right
                can_go_right[row, col] = img_gray[cy, wall_x] > 128

            if row < N - 1:
                wall_y = row * CELL_PX + CELL_PX   # first boundary pixel below
                can_go_down[row, col] = img_gray[wall_y, cx] > 128

    return can_go_right, can_go_down


# ─── border opening detection (Start / Goal) ─────────────────────────────────

def detect_border_openings(img_gray: np.ndarray) -> dict:
    """
    Scan the 2px outer border for white gaps — these are the maze entrance/exit.

    Returns:
      {
        'top':    list of col indices with opening on the top border,
        'bottom': list of col indices with opening on the bottom border,
        'left':   list of row indices with opening on the left border,
        'right':  list of row indices with opening on the right border,
      }
    """
    N   = GRID_SIZE
    IMG = img_gray.shape[0]   # = 1026

    openings: dict = {"top": [], "bottom": [], "left": [], "right": []}

    for col in range(N):
        x0, x1 = col * CELL_PX + BORDER, col * CELL_PX + CELL_PX
        if img_gray[0:BORDER, x0:x1].max() > 128:
            openings["top"].append(col)
        if img_gray[IMG - BORDER:IMG, x0:x1].max() > 128:
            openings["bottom"].append(col)

    for row in range(N):
        y0, y1 = row * CELL_PX + BORDER, row * CELL_PX + CELL_PX
        if img_gray[y0:y1, 0:BORDER].max() > 128:
            openings["left"].append(row)
        if img_gray[y0:y1, IMG - BORDER:IMG].max() > 128:
            openings["right"].append(row)

    return openings


# ─── hazard / special cell detection ─────────────────────────────────────────

def _cell_avg_color(img_rgb: np.ndarray, col: int, row: int) -> tuple[float, float, float, int]:
    """
    Average color of non-white pixels inside the cell interior.
    Returns (R, G, B, npx).  npx=0 means the cell is plain white (no marker).
    """
    y0 = row * CELL_PX + BORDER
    y1 = row * CELL_PX + CELL_PX        # exclusive
    x0 = col * CELL_PX + BORDER
    x1 = col * CELL_PX + CELL_PX        # exclusive

    cell = img_rgb[y0:y1, x0:x1].astype(np.int32)
    # non-white: at least one channel deviates from 255 by more than 30
    mask = np.any(np.abs(cell - 255) > 30, axis=2)
    # exclude black wall bleed pixels (all channels near 0)
    not_black = np.any(cell > 30, axis=2)
    mask &= not_black

    npx = int(mask.sum())
    if npx == 0:
        return 0.0, 0.0, 0.0, 0

    avg = cell[mask].mean(axis=0)
    return float(avg[0]), float(avg[1]), float(avg[2]), npx


def _classify_color(r: float, g: float, b: float, npx: int) -> int:
    """
    Map average cell color → cell type.

    Observed average colors in maze-alpha/MAZE_1.png:
    Teleport pairs are color-coded (same color = pad ↔ destination):
      🟢 Green  pair: R≈103-125, G≈213-218, B≈154-165   npx≈174-181
      🟣 Purple pair: R≈140-152, G≈113-116, B≈194-224   npx≈163-184
      🟡 Gold   pair: R≈233-255, G≈162-190, B≈92-98     npx≈174
      💀 Skull (confusion):  R≈197-200, G≈138-146, B≈79-82  npx≈168-177
      🔥 Fire  (death pit):  R≈245,     G≈146,     B≈82      npx≈46-82

    Key discriminators:
      - Gold vs Fire: both have high R, but Gold has G > 155 while Fire has G ≈ 145.
      - Green teleport vs Confusion skull: G dominant (G > R+60) for green markers.
      - Purple: B clearly highest channel.
    """
    if npx < 4:
        return CELL_EMPTY

    # Purple teleport: B clearly dominant
    if b > 170 and b > r + 40 and b > g + 50:
        return CELL_TELEPORT

    # Green teleport: G clearly dominant (both solid circle and snowflake variants)
    if g > 180 and g > r + 60:
        return CELL_TELEPORT

    # Gold teleport: warm orange circle — distinguished from fire by G > 155
    if r > 215 and g > 155 and b < 130:
        return CELL_TELEPORT

    # Skull emoji (confusion): orange-brown, R < 215, moderate G
    if r > 170 and r < 215 and g > 120 and g < 160 and b < 100:
        return CELL_CONFUSION

    # Fire emoji (death pit): bright orange, R > 215, G ≤ 155
    if r > 215 and g > 100 and b < 110 and (r - b) > 130:
        return CELL_PIT

    return CELL_EMPTY


def extract_cell_types(img_rgb: np.ndarray) -> np.ndarray:
    """Returns a (64, 64) int array with cell type codes."""
    N = GRID_SIZE
    types = np.zeros((N, N), dtype=np.int32)
    for row in range(N):
        for col in range(N):
            r, g, b, npx = _cell_avg_color(img_rgb, col, row)
            t = _classify_color(r, g, b, npx)
            if t != CELL_EMPTY:
                types[row, col] = t
    return types


# ─── main parse function ──────────────────────────────────────────────────────

def parse_maze(maze_dir: str | Path) -> dict:
    """
    Parse a maze directory containing MAZE_0.png and MAZE_1.png.

    Returns a dict with:
      'cell_type'      : np.ndarray (64,64) int  — cell type per cell
      'can_go_right'   : np.ndarray (64,64) bool — True = passage to the right
      'can_go_down'    : np.ndarray (64,64) bool — True = passage below
      'border_openings': dict with 'top'/'bottom'/'left'/'right' lists of open cols/rows
      'start'          : (col, row) — top border opening (agent spawn)
      'goal'           : (col, row) — bottom border opening (exit)
      'pits'           : list of (col, row)
      'teleports'  : list of (col, row)  — all teleport markers (pad + dest, color-paired)
      'confusions' : list of (col, row)
    """
    maze_dir = Path(maze_dir)
    img0_gray = np.array(Image.open(maze_dir / "MAZE_0.png").convert("L"))
    img1_rgb  = np.array(Image.open(maze_dir / "MAZE_1.png").convert("RGB"))

    can_go_right, can_go_down = extract_walls(img0_gray)
    cell_type = extract_cell_types(img1_rgb)
    openings  = detect_border_openings(img0_gray)

    # Agent enters from the bottom, exits through the top
    start = (openings["bottom"][0], GRID_SIZE - 1)    if openings["bottom"] else None
    goal  = (openings["top"][0],    0)                if openings["top"]    else None

    # Stamp Start / Goal into cell_type array
    if start:
        cell_type[start[1], start[0]] = CELL_START
    if goal:
        cell_type[goal[1],  goal[0]]  = CELL_GOAL

    def _rc_to_cr(arr):
        rows, cols = np.where(arr)
        return [(int(c), int(r)) for r, c in zip(rows, cols)]

    return {
        "cell_type":       cell_type,
        "can_go_right":    can_go_right,
        "can_go_down":     can_go_down,
        "border_openings": openings,
        "start":           start,
        "goal":            goal,
        "pits":            _rc_to_cr(cell_type == CELL_PIT),
        "teleports":   _rc_to_cr(cell_type == CELL_TELEPORT),
        "confusions":  _rc_to_cr(cell_type == CELL_CONFUSION),
    }


# ─── quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for maze_id, maze_dir in [("alpha (training)", "maze-alpha"), ("gamma (testing)", "maze-gamma")]:
        print(f"\n{'='*50}")
        print(f"Maze: {maze_id}")
        print(f"{'='*50}")

        data = parse_maze(maze_dir)

        print(f"  Start:           {data['start']}  (border opening)")
        print(f"  Goal:            {data['goal']}  (border opening)")
        print(f"  Pits:            {len(data['pits'])} cells")
        print(f"  Teleports:  {len(data['teleports'])} cells  → {data['teleports']}")
        print(f"  Confusions: {len(data['confusions'])} cells → {data['confusions']}")
        print(f"  Border openings: {data['border_openings']}")
        print(f"  Open RIGHT passages: {data['can_go_right'].sum()}")
        print(f"  Open DOWN  passages: {data['can_go_down'].sum()}")

        # Verify wall connectivity: every cell should be reachable from start
        from collections import deque
        start = data['start']
        if start:
            visited = set()
            q = deque([start])
            visited.add(start)
            right = data['can_go_right']
            down  = data['can_go_down']
            while q:
                col, row = q.popleft()
                for dc, dr, arr in [
                    ( 1, 0, right), (-1, 0, right),
                    ( 0, 1, down),  ( 0,-1, down),
                ]:
                    nc, nr = col+dc, row+dr
                    if (nc, nr) in visited or not (0 <= nc < 64 and 0 <= nr < 64):
                        continue
                    # check passage in the correct direction
                    if dc == 1  and right[row, col]:      visited.add((nc,nr)); q.append((nc,nr))
                    elif dc == -1 and right[row, nc]:     visited.add((nc,nr)); q.append((nc,nr))
                    elif dr == 1  and down[row, col]:     visited.add((nc,nr)); q.append((nc,nr))
                    elif dr == -1 and down[nr, col]:      visited.add((nc,nr)); q.append((nc,nr))
            print(f"  BFS from start reaches: {len(visited)} / 4096 cells")
