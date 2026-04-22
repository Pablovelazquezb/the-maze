"""
Maze Environment — COSC 4368 Spring 2026
Supports: walls, death pits (fire with rotation), teleporters, confusion pads
Fire/death pits rotate 90° clockwise every 5 actions (pivot = bottom of V shape).
"""

from typing import List, Tuple, Dict, Optional
from enum import Enum
from collections import deque
import random, math

# ── Actions ───────────────────────────────────────────────────────────────────
class Action(Enum):
    MOVE_UP    = 0
    MOVE_DOWN  = 1
    MOVE_LEFT  = 2
    MOVE_RIGHT = 3
    WAIT       = 4

ACTION_DELTA = {
    Action.MOVE_UP:    ( 0, -1),
    Action.MOVE_DOWN:  ( 0,  1),
    Action.MOVE_LEFT:  (-1,  0),
    Action.MOVE_RIGHT: ( 1,  0),
    Action.WAIT:       ( 0,  0),
}

CONFUSED_MAP = {
    Action.MOVE_UP:    Action.MOVE_DOWN,
    Action.MOVE_DOWN:  Action.MOVE_UP,
    Action.MOVE_LEFT:  Action.MOVE_RIGHT,
    Action.MOVE_RIGHT: Action.MOVE_LEFT,
    Action.WAIT:       Action.WAIT,
}

# ── Turn Result ───────────────────────────────────────────────────────────────
class TurnResult:
    def __init__(self):
        self.wall_hits: int = 0
        self.current_position: Tuple[int, int] = (0, 0)
        self.is_dead: bool = False
        self.is_confused: bool = False
        self.is_goal_reached: bool = False
        self.teleported: bool = False
        self.actions_executed: int = 0

# ── Agent base class ─────────────────────────────────────────────────────────
class Agent:
    def __init__(self):
        self.memory = {}
    def plan_turn(self, last_result: TurnResult) -> List[Action]:
        raise NotImplementedError
    def reset_episode(self):
        pass

# ── Maze Environment ─────────────────────────────────────────────────────────
class MazeEnvironment:
    def __init__(self, maze_id: str, max_turns: int = 10000):
        self.maze_id = maze_id
        self.max_turns = max_turns

        with open(maze_id, 'r') as f:
            self.text_maze = [list(line.rstrip('\n')) for line in f.readlines()]

        self.start_pos = None
        self.goal = None
        self.teleporters: List[Tuple[int,int]] = []
        self.pits: List[Tuple[int,int]] = []
        self.confusion_pads: List[Tuple[int,int]] = []

        # Parse cells
        for y in range(64):
            for x in range(64):
                cell = self._get_cell(x, y)
                if cell == 'S':
                    if self.start_pos is None:
                        self.start_pos = (x, y)
                elif cell == 'G':
                    self.goal = (x, y)
                elif cell == 'T':
                    self.teleporters.append((x, y))
                elif cell == 'P':
                    self.pits.append((x, y))
                elif cell == 'C':
                    self.confusion_pads.append((x, y))

        # Find largest connected component for injection
        largest_cc = self._find_largest_cc()

        rng = random.Random(42)

        # Inject Start if missing
        if self.start_pos is None:
            if largest_cc:
                self.start_pos = largest_cc[0]
                self._set_cell(*self.start_pos, 'S')

        # Inject Goal if missing
        if self.goal is None:
            if largest_cc:
                self.goal = largest_cc[-1]
                self._set_cell(*self.goal, 'G')

        # Inject hazards for maze_0 / maze_1 / maze_2 if missing
        if True:  # Inject hazards for all mazes if missing
            avail = [c for c in largest_cc if c != self.start_pos and c != self.goal]
            if len(self.pits) == 0:
                for _ in range(10):
                    if avail:
                        pos = avail.pop(rng.randint(0, len(avail)-1))
                        self.pits.append(pos)
                        self._set_cell(*pos, 'P')
            if len(self.teleporters) == 0:
                for _ in range(6):
                    if avail:
                        pos = avail.pop(rng.randint(0, len(avail)-1))
                        self.teleporters.append(pos)
                        self._set_cell(*pos, 'T')
            if len(self.confusion_pads) == 0:
                for _ in range(5):
                    if avail:
                        pos = avail.pop(rng.randint(0, len(avail)-1))
                        self.confusion_pads.append(pos)
                        self._set_cell(*pos, 'C')

        # Pair teleporters
        random.seed(42)
        self.teleport_map: Dict[Tuple, Tuple] = {}
        if self.teleporters:
            sources = list(self.teleporters)
            dests = list(self.teleporters)
            random.shuffle(dests)
            for i in range(len(sources)):
                if sources[i] == dests[i] and len(sources) > 1:
                    swap_idx = (i + 1) % len(sources)
                    dests[i], dests[swap_idx] = dests[swap_idx], dests[i]
            for s, d in zip(sources, dests):
                self.teleport_map[s] = d

        # Fire rotation state: direction index (0=UP,1=RIGHT,2=DOWN,3=LEFT)
        self.pit_directions: Dict[Tuple, int] = {}
        for p in self.pits:
            self.pit_directions[p] = 0  # initial direction = UP

        self.total_actions_taken = 0
        self.reset()

    # ── Internal helpers ──────────────────────────────────────────────────────
    def _get_cell(self, x: int, y: int) -> str:
        if x < 0 or x >= 64 or y < 0 or y >= 64:
            return 'X'
        return self.text_maze[y*2+1][x*2+1]

    def _set_cell(self, x: int, y: int, char: str):
        if 0 <= x < 64 and 0 <= y < 64:
            self.text_maze[y*2+1][x*2+1] = char

    def _has_wall(self, x: int, y: int, direction: Action) -> bool:
        if direction == Action.MOVE_UP:
            return y == 0 or self.text_maze[y*2][x*2+1] == 'X'
        elif direction == Action.MOVE_DOWN:
            return y == 63 or self.text_maze[y*2+2][x*2+1] == 'X'
        elif direction == Action.MOVE_LEFT:
            return x == 0 or self.text_maze[y*2+1][x*2] == 'X'
        elif direction == Action.MOVE_RIGHT:
            return x == 63 or self.text_maze[y*2+1][x*2+2] == 'X'
        return False

    def _find_largest_cc(self):
        visited = set()
        largest = []
        for sy in range(64):
            for sx in range(64):
                cell = self._get_cell(sx, sy)
                if (sx, sy) not in visited and cell in ('O', 'S', 'G'):
                    q = deque([(sx, sy)])
                    cc = []
                    visited.add((sx, sy))
                    while q:
                        cx, cy = q.popleft()
                        cc.append((cx, cy))
                        for a in [Action.MOVE_UP, Action.MOVE_DOWN, Action.MOVE_LEFT, Action.MOVE_RIGHT]:
                            dx, dy = ACTION_DELTA[a]
                            nx, ny = cx + dx, cy + dy
                            if 0 <= nx < 64 and 0 <= ny < 64 and (nx, ny) not in visited:
                                if not self._has_wall(cx, cy, a):
                                    visited.add((nx, ny))
                                    q.append((nx, ny))
                        if len(cc) > len(largest):
                            largest = cc
        return largest

    def _is_pit(self, x, y):
        return (x, y) in self.pit_directions

    def _rotate_fires(self):
        """Rotate all fire pits 90° CW and move them one step in their direction."""
        DIR_ACTIONS = [Action.MOVE_UP, Action.MOVE_RIGHT, Action.MOVE_DOWN, Action.MOVE_LEFT]
        new_pits = {}
        pits_set_old = set(self.pit_directions.keys())

        for (px, py), dir_idx in list(self.pit_directions.items()):
            new_dir = (dir_idx + 1) % 4
            move_action = DIR_ACTIONS[dir_idx]
            dx, dy = ACTION_DELTA[move_action]
            nx, ny = px + dx, py + dy

            if (0 <= nx < 64 and 0 <= ny < 64 and
                not self._has_wall(px, py, move_action) and
                (nx, ny) not in new_pits):
                new_pits[(nx, ny)] = new_dir
            else:
                new_pits[(px, py)] = new_dir

        # Update pit tracking
        self.pit_directions = new_pits
        # Update the pits list
        self.pits = list(new_pits.keys())

    # ── Public API ────────────────────────────────────────────────────────────
    def reset(self) -> Tuple[int, int]:
        self.current_pos = self.start_pos
        self.turns_taken = 0
        self.deaths = 0
        self.confused_count = 0
        self.cells_explored = set([self.start_pos])
        self.is_confused = False
        self.confusion_turns_left = 0
        self.goal_reached = False
        self.total_actions_taken = 0
        # Reset fire positions
        self.pit_directions = {}
        for p in self.pits:
            self.pit_directions[p] = 0
        return self.current_pos

    def step(self, actions: List[Action]) -> TurnResult:
        if not actions or len(actions) > 5:
            raise ValueError("Must provide 1-5 actions per turn")

        result = TurnResult()

        # Handle confusion carry-over
        if self.confusion_turns_left > 0:
            self.confusion_turns_left -= 1
            if self.confusion_turns_left == 0:
                self.is_confused = False

        for act in actions:
            result.actions_executed += 1
            self.total_actions_taken += 1

            # Fire rotation every 5 actions
            if self.total_actions_taken > 0 and self.total_actions_taken % 5 == 0:
                self._rotate_fires()

            # Confusion inversion
            effective = CONFUSED_MAP[act] if self.is_confused else act

            x, y = self.current_pos

            if effective == Action.WAIT:
                pass
            elif effective in (Action.MOVE_UP, Action.MOVE_DOWN, Action.MOVE_LEFT, Action.MOVE_RIGHT):
                if self._has_wall(x, y, effective):
                    result.wall_hits += 1
                else:
                    dx, dy = ACTION_DELTA[effective]
                    x, y = x + dx, y + dy
                    self.current_pos = (x, y)
                    self.cells_explored.add((x, y))

            # Check cell effects
            if self._is_pit(x, y):
                result.is_dead = True
                self.deaths += 1
                self.current_pos = self.start_pos
                break

            if (x, y) == self.goal:
                result.is_goal_reached = True
                self.goal_reached = True
                break

            if (x, y) in self.teleport_map:
                result.teleported = True
                self.current_pos = self.teleport_map[(x, y)]
                x, y = self.current_pos
                self.cells_explored.add((x, y))

            if (x, y) in [(cx, cy) for cx, cy in self.confusion_pads]:
                result.is_confused = True
                self.is_confused = True
                self.confused_count += 1
                self.confusion_turns_left = 1

        result.current_position = self.current_pos
        self.turns_taken += 1
        return result

    def get_episode_stats(self) -> dict:
        return {
            'turns_taken': self.turns_taken,
            'deaths': self.deaths,
            'confused': self.confused_count,
            'cells_explored': len(self.cells_explored),
            'goal_reached': self.goal_reached
        }
