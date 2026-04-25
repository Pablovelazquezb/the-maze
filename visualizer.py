"""
Maze Visualizer
Renders the maze, the agent's known map, and episode replays using matplotlib.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from maze_parser import (
    CELL_CONFUSION, CELL_GOAL, CELL_PIT, CELL_START,
    CELL_TELEPORT, parse_maze,
)

# ─── color palette ────────────────────────────────────────────────────────────

COLORS = {
    "bg_known":    "#FFFFFF",
    "bg_unknown":  "#C8C8C8",
    "wall":        "#1A1A1A",
    "start":       "#4CAF50",
    "goal":        "#FFC107",
    "pit":         "#F44336",
    "teleport":    "#9C27B0",
    "confusion":   "#00BCD4",
    "agent":       "#2196F3",
    "path":        "#FF9800",
}

CELL_COLORS = {
    CELL_START:     COLORS["start"],
    CELL_GOAL:      COLORS["goal"],
    CELL_PIT:       COLORS["pit"],
    CELL_TELEPORT:  COLORS["teleport"],
    CELL_CONFUSION: COLORS["confusion"],
}

CELL_LABELS = {
    CELL_START:     "S",
    CELL_GOAL:      "G",
    CELL_PIT:       "P",
    CELL_TELEPORT:  "T",
    CELL_CONFUSION: "C",
}


# ─── core drawing helpers ─────────────────────────────────────────────────────

def _build_wall_segments(
    can_go_right:   np.ndarray,
    can_go_down:    np.ndarray,
    border_openings: Optional[dict] = None,
) -> list:
    """
    Return wall segments as ((x0,y0),(x1,y1)) pairs.
    border_openings can suppress gaps in the outer border at entrance/exit.
    """
    N = 64
    op = border_openings or {}
    top_open    = set(op.get("top",    []))
    bottom_open = set(op.get("bottom", []))
    left_open   = set(op.get("left",   []))
    right_open  = set(op.get("right",  []))

    segments = []

    # ── outer border (drawn cell-by-cell so we can skip openings) ────────────
    for col in range(N):
        if col not in top_open:
            segments.append(((col, 0), (col + 1, 0)))           # top edge
        if col not in bottom_open:
            segments.append(((col, N), (col + 1, N)))           # bottom edge

    for row in range(N):
        if row not in left_open:
            segments.append(((0, row), (0, row + 1)))           # left edge
        if row not in right_open:
            segments.append(((N, row), (N, row + 1)))           # right edge

    # ── internal walls ────────────────────────────────────────────────────────
    for row in range(N):
        for col in range(N):
            if col < N - 1 and not can_go_right[row, col]:
                segments.append(((col + 1, row), (col + 1, row + 1)))
            if row < N - 1 and not can_go_down[row, col]:
                segments.append(((col, row + 1), (col + 1, row + 1)))

    return segments


def _draw_walls(ax, can_go_right, can_go_down,
                border_openings=None, color="#1A1A1A", lw=1.2):
    segs = _build_wall_segments(can_go_right, can_go_down, border_openings)
    ax.add_collection(LineCollection(segs, colors=color, linewidths=lw, zorder=3))


def _draw_cell(ax, col, row, color, alpha=1.0, zorder=2):
    ax.add_patch(plt.Rectangle(
        (col, row), 1, 1,
        facecolor=color, edgecolor="none", alpha=alpha, zorder=zorder,
    ))


def _draw_special_cells(ax, cell_type: np.ndarray, show_labels=True):
    for ctype, color in CELL_COLORS.items():
        rows, cols = np.where(cell_type == ctype)
        for r, c in zip(rows, cols):
            _draw_cell(ax, c, r, color, zorder=2)
            if show_labels:
                ax.text(c + 0.5, r + 0.5, CELL_LABELS[ctype],
                        ha="center", va="center",
                        fontsize=4, fontweight="bold", color="white", zorder=5)


def _finish_ax(ax, title, N=64):
    ax.set_xlim(0, N)
    ax.set_ylim(0, N)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")
    ax.set_title(title, fontsize=11, pad=6)


def _legend(ax, items: List[Tuple[str, str]]):
    patches = [mpatches.Patch(color=c, label=l) for l, c in items]
    ax.legend(handles=patches, loc="lower right", fontsize=6,
              framealpha=0.8, edgecolor="gray")


# ─── public API ───────────────────────────────────────────────────────────────

def visualize_maze(
    maze_dir: str,
    title: str = "Maze",
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Ground-truth view: full maze with all hazards labeled."""
    data = parse_maze(maze_dir)

    _, ax = plt.subplots(figsize=(9, 9))
    ax.set_facecolor(COLORS["bg_known"])

    _draw_special_cells(ax, data["cell_type"])
    _draw_walls(ax, data["can_go_right"], data["can_go_down"],
                border_openings=data["border_openings"])
    _finish_ax(ax, title)
    _legend(ax, [
        ("Start",     COLORS["start"]),
        ("Goal",      COLORS["goal"]),
        ("Death pit", COLORS["pit"]),
        ("Teleport",  COLORS["teleport"]),
        ("Confusion", COLORS["confusion"]),
    ])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    if show:
        plt.show()
    plt.close()


def visualize_agent_map(
    maze_dir: str,
    known_cells: Set[Tuple[int, int]],
    agent_pos: Optional[Tuple[int, int]] = None,
    known_hazards: Optional[Dict[Tuple[int, int], int]] = None,
    path: Optional[List[Tuple[int, int]]] = None,
    title: str = "Agent's Known Map",
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Agent's perspective: fog of war over unexplored cells.
    known_hazards: {(col,row): cell_type} discovered during exploration.
    path: planned route to highlight.
    """
    data = parse_maze(maze_dir)

    _, ax = plt.subplots(figsize=(9, 9))
    ax.set_facecolor(COLORS["bg_unknown"])

    for (col, row) in known_cells:
        _draw_cell(ax, col, row, COLORS["bg_known"], zorder=1)

    if path:
        for (col, row) in path:
            _draw_cell(ax, col, row, COLORS["path"], alpha=0.4, zorder=2)

    if known_hazards:
        for (col, row), ctype in known_hazards.items():
            _draw_cell(ax, col, row, CELL_COLORS.get(ctype, COLORS["bg_known"]), zorder=2)
            ax.text(col + 0.5, row + 0.5, CELL_LABELS.get(ctype, "?"),
                    ha="center", va="center",
                    fontsize=4, fontweight="bold", color="white", zorder=5)

    if agent_pos:
        c, r = agent_pos
        ax.add_patch(plt.Circle((c + 0.5, r + 0.5), 0.35,
                                color=COLORS["agent"], zorder=6))

    _draw_walls(ax, data["can_go_right"], data["can_go_down"],
                border_openings=data["border_openings"], color="#666666", lw=0.8)
    _finish_ax(ax, title)
    _legend(ax, [
        ("Explored",  COLORS["bg_known"]),
        ("Unknown",   COLORS["bg_unknown"]),
        ("Agent",     COLORS["agent"]),
        ("Path",      COLORS["path"]),
        ("Death pit", COLORS["pit"]),
        ("Teleport",  COLORS["teleport"]),
        ("Confusion", COLORS["confusion"]),
    ])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    if show:
        plt.show()
    plt.close()


def visualize_episode_path(
    maze_dir: str,
    positions: List[Tuple[int, int]],
    deaths: Optional[List[Tuple[int, int]]] = None,
    title: str = "Episode Path",
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Heatmap of a full episode trajectory.
    Darker = cell visited more times. Death spots marked with ✕.
    """
    data = parse_maze(maze_dir)
    N    = 64

    visit_count = np.zeros((N, N), dtype=int)
    for (col, row) in positions:
        visit_count[row, col] += 1

    _, ax = plt.subplots(figsize=(9, 9))
    ax.set_facecolor("#F5F5F5")

    max_v = max(int(visit_count.max()), 1)
    cmap  = plt.cm.YlOrRd
    for row in range(N):
        for col in range(N):
            v = visit_count[row, col]
            if v > 0:
                _draw_cell(ax, col, row, cmap(0.15 + 0.85 * v / max_v), zorder=1)

    _draw_special_cells(ax, data["cell_type"], show_labels=True)

    if positions:
        sc, sr = positions[0]
        ec, er = positions[-1]
        ax.plot(sc + 0.5, sr + 0.5, "o", color=COLORS["agent"], markersize=8, zorder=7)
        ax.plot(ec + 0.5, er + 0.5, "*", color=COLORS["agent"], markersize=12, zorder=7)

    if deaths:
        for (col, row) in deaths:
            ax.text(col + 0.5, row + 0.5, "✕",
                    ha="center", va="center",
                    fontsize=7, color="black", fontweight="bold", zorder=8)

    _draw_walls(ax, data["can_go_right"], data["can_go_down"],
                border_openings=data["border_openings"])
    _finish_ax(ax, title)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    if show:
        plt.show()
    plt.close()


def visualize_learning_curves(
    history: List[Dict],
    title: str = "Training Progress",
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Plot turns, deaths, cells explored, and success rate over training episodes."""
    if not history:
        print("No history to plot.")
        return

    episodes = list(range(1, len(history) + 1))
    turns    = [h.get("turns_taken", 0)   for h in history]
    deaths   = [h.get("deaths", 0)         for h in history]
    explored = [h.get("cells_explored", 0) for h in history]
    success  = [1 if h.get("goal_reached") else 0 for h in history]

    def _smooth(data, w=10):
        if len(data) < w:
            return data
        return np.convolve(data, np.ones(w) / w, mode="same").tolist()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=13)

    for ax, values, subtitle, color, ylabel in [
        (axes[0, 0], turns,   "Turns per Episode",     COLORS["agent"],    "Turns"),
        (axes[0, 1], deaths,  "Deaths per Episode",    COLORS["pit"],      "Deaths"),
        (axes[1, 0], explored,"Cells Explored",        COLORS["teleport"], "Cells"),
        (axes[1, 1], success, "Success Rate (rolling)",COLORS["goal"],     "Success"),
    ]:
        ax.plot(episodes, values, color=color, alpha=0.3, linewidth=0.8)
        ax.plot(episodes, _smooth(values), color=color, linewidth=2)
        ax.set_title(subtitle, fontsize=10)
        ax.set_xlabel("Episode", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    if show:
        plt.show()
    plt.close()


# ─── demo ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    from collections import deque
    os.makedirs("output", exist_ok=True)

    print("1. Ground truth — alpha...")
    visualize_maze("maze-alpha", title="Training Maze — Ground Truth",
                   save_path="output/maze_alpha_ground_truth.png", show=False)

    print("2. Ground truth — gamma...")
    visualize_maze("maze-gamma", title="Testing Maze — Ground Truth",
                   save_path="output/maze_gamma_ground_truth.png", show=False)

    print("3. Agent map (300 cells explored)...")
    data  = parse_maze("maze-alpha")
    cr, cd, start = data["can_go_right"], data["can_go_down"], data["start"]
    ct    = data["cell_type"]

    visited, q = {start}, deque([start])
    while q and len(visited) < 300:
        col, row = q.popleft()
        for dc, dr, ok in [
            ( 1, 0, cr[row, col]),
            (-1, 0, col > 0 and cr[row, col - 1]),
            ( 0, 1, cd[row, col]),
            ( 0,-1, row > 0 and cd[row - 1, col]),
        ]:
            nc, nr = col + dc, row + dr
            if ok and 0 <= nc < 64 and 0 <= nr < 64 and (nc, nr) not in visited:
                visited.add((nc, nr))
                q.append((nc, nr))

    known_hazards = {
        (c, r): int(ct[r, c])
        for (c, r) in visited
        if ct[r, c] in (CELL_PIT, CELL_CONFUSION)
    }
    visualize_agent_map("maze-alpha", known_cells=visited,
                        agent_pos=list(visited)[-1], known_hazards=known_hazards,
                        title="Agent Map — 300 cells explored",
                        save_path="output/agent_map_demo.png", show=False)

    print("4. Episode path (BFS to goal)...")
    goal   = data["goal"]
    parent = {start: None}
    q2     = deque([start])
    while q2:
        col, row = q2.popleft()
        if (col, row) == goal:
            break
        for dc, dr, ok in [
            ( 1, 0, cr[row, col]),
            (-1, 0, col > 0 and cr[row, col - 1]),
            ( 0, 1, cd[row, col]),
            ( 0,-1, row > 0 and cd[row - 1, col]),
        ]:
            nc, nr = col + dc, row + dr
            if ok and 0 <= nc < 64 and 0 <= nr < 64 and (nc, nr) not in parent:
                parent[(nc, nr)] = (col, row)
                q2.append((nc, nr))

    path, node = [], goal
    while node:
        path.append(node)
        node = parent.get(node)
    path.reverse()

    visualize_episode_path("maze-alpha", positions=path,
                           title=f"Optimal BFS Path — {len(path)} steps",
                           save_path="output/episode_path_demo.png", show=False)

    print("\nImágenes en output/")
