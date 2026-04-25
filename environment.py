"""
MazeEnvironment — local simulation of the maze API defined in the project spec.

The instructor provides the real environment server; this module lets us test
the agent locally using the parsed PNG data.  The interface is identical to
the spec so the agent code can run unmodified against either.

Teleport destinations are not encoded in the PNG images.  They must be supplied
in TELEPORT_DESTINATIONS below (or overridden at construction time) and will be
discovered by the agent through exploration at runtime.
"""

from __future__ import annotations

from collections import defaultdict
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from maze_parser import parse_maze, CELL_PIT, CELL_TELEPORT, CELL_CONFUSION, CELL_GOAL


# ─── known teleport destinations ─────────────────────────────────────────────
# Format: { maze_id: { (src_col, src_row): (dst_col, dst_row) } }
# Destinations are navigable, non-pit cells discovered by looking at the maze.
# Update these once the agent discovers the real values.

TELEPORT_DESTINATIONS: Dict[str, Dict[Tuple[int,int], Tuple[int,int]]] = {
    "training": {
        (9,  46): (40, 20),   # placeholder — update after first run
        (26, 54): (15, 10),   # placeholder — update after first run
    },
    "testing": {
        (27, 13): (50, 50),
        (38, 17): (20, 40),
        (10, 21): (45, 10),
        (33, 23): (5,  55),
        ( 8, 49): (60, 5),
        (55, 49): (3,  30),
        (45, 56): (30, 3),
        ( 0, 63): (63, 0),
    },
}


# ─── public data types (spec §6.1) ───────────────────────────────────────────

class Action(Enum):
    MOVE_UP    = 0
    MOVE_DOWN  = 1
    MOVE_LEFT  = 2
    MOVE_RIGHT = 3
    WAIT       = 4


class TurnResult:
    def __init__(self) -> None:
        self.wall_hits:       int              = 0
        self.current_position: Tuple[int, int] = (0, 0)
        self.is_dead:         bool             = False
        self.is_confused:     bool             = False
        self.is_goal_reached: bool             = False
        self.teleported:      bool             = False
        self.actions_executed: int             = 0

    def __repr__(self) -> str:
        return (
            f"TurnResult(pos={self.current_position}, "
            f"dead={self.is_dead}, goal={self.is_goal_reached}, "
            f"wall_hits={self.wall_hits}, confused={self.is_confused}, "
            f"teleported={self.teleported}, executed={self.actions_executed})"
        )


# ─── environment ─────────────────────────────────────────────────────────────

class MazeEnvironment:
    """
    Local simulation of the maze environment.

    Maze coordinate system (from spec):
      (0,0) = top-left,  (63,63) = bottom-right
      col = x-axis,      row = y-axis
      MOVE_UP   → row - 1
      MOVE_DOWN → row + 1
    """

    MAZE_DIRS = {
        "training": "maze-alpha",
        "testing":  "maze-gamma",
    }

    def __init__(
        self,
        maze_id: str,
        teleport_destinations: Optional[Dict[Tuple[int,int], Tuple[int,int]]] = None,
    ) -> None:
        if maze_id not in self.MAZE_DIRS:
            raise ValueError(f"maze_id must be 'training' or 'testing', got '{maze_id}'")

        data = parse_maze(self.MAZE_DIRS[maze_id])

        self._cell_type:    np.ndarray = data["cell_type"]      # (64,64)
        self._can_go_right: np.ndarray = data["can_go_right"]   # (64,64) bool
        self._can_go_down:  np.ndarray = data["can_go_down"]    # (64,64) bool
        self._start:        Tuple[int,int] = data["start"]
        self._goal:         Tuple[int,int] = data["goal"]

        self._teleport_map: Dict[Tuple[int,int], Tuple[int,int]] = (
            teleport_destinations
            if teleport_destinations is not None
            else TELEPORT_DESTINATIONS.get(maze_id, {})
        )

        # episode state
        self._pos:           Tuple[int,int] = self._start
        self._turn:          int = 0
        self._deaths:        int = 0
        self._confused_count: int = 0
        self._cells_visited: set = set()
        self._goal_reached:  bool = False

        # confusion lasts rest of current turn + next full turn
        # _confused_turns_left: how many MORE turns the inversion applies
        self._confused_turns_left: int = 0
        self._confused_this_turn:  bool = False   # triggered mid-turn

    # ── passage helpers ────────────────────────────────────────────────────────

    def _can_move(self, col: int, row: int, action: Action) -> bool:
        """True if the move is within bounds and no wall blocks it."""
        N = 64
        if action == Action.MOVE_RIGHT:
            return col < N - 1 and bool(self._can_go_right[row, col])
        if action == Action.MOVE_LEFT:
            return col > 0     and bool(self._can_go_right[row, col - 1])
        if action == Action.MOVE_DOWN:
            return row < N - 1 and bool(self._can_go_down[row, col])
        if action == Action.MOVE_UP:
            return row > 0     and bool(self._can_go_down[row - 1, col])
        return True  # WAIT always succeeds

    @staticmethod
    def _apply_confusion(action: Action) -> Action:
        inversions = {
            Action.MOVE_UP:    Action.MOVE_DOWN,
            Action.MOVE_DOWN:  Action.MOVE_UP,
            Action.MOVE_LEFT:  Action.MOVE_RIGHT,
            Action.MOVE_RIGHT: Action.MOVE_LEFT,
        }
        return inversions.get(action, action)

    @staticmethod
    def _delta(action: Action) -> Tuple[int, int]:
        """Returns (dcol, drow) for the action."""
        return {
            Action.MOVE_RIGHT: ( 1,  0),
            Action.MOVE_LEFT:  (-1,  0),
            Action.MOVE_DOWN:  ( 0,  1),
            Action.MOVE_UP:    ( 0, -1),
            Action.WAIT:       ( 0,  0),
        }[action]

    # ── public API ─────────────────────────────────────────────────────────────

    def reset(self) -> Tuple[int, int]:
        """Reset for a new episode. Returns starting position."""
        self._pos              = self._start
        self._turn             = 0
        self._deaths           = 0
        self._confused_count   = 0
        self._cells_visited    = {self._start}
        self._goal_reached     = False
        self._confused_turns_left = 0
        self._confused_this_turn  = False
        return self._start

    def step(self, actions: List[Action]) -> TurnResult:
        """
        Execute up to 5 actions sequentially.

        Confusion mechanic (spec §4.4):
          - Touching a confusion cell sets confusion for rest of this turn
            AND the entire following turn.
          - While confused, UP↔DOWN and LEFT↔RIGHT are swapped.
        """
        if not actions:
            raise ValueError("actions list cannot be empty")
        if len(actions) > 5:
            raise ValueError("actions list cannot have more than 5 actions")

        result = TurnResult()
        col, row = self._pos

        # Confusion carries over from previous turn
        is_confused = self._confused_turns_left > 0
        self._confused_this_turn = False

        for i, raw_action in enumerate(actions):
            action = self._apply_confusion(raw_action) if is_confused else raw_action

            if action == Action.WAIT:
                result.actions_executed += 1
                continue

            if not self._can_move(col, row, action):
                result.wall_hits += 1
                result.actions_executed += 1
                continue

            # Move is valid — update position
            dcol, drow = self._delta(action)
            col += dcol
            row += drow
            self._cells_visited.add((col, row))
            result.actions_executed += 1

            cell = int(self._cell_type[row, col])

            # ── teleport ──
            while cell == CELL_TELEPORT and (col, row) in self._teleport_map:
                dest = self._teleport_map[(col, row)]
                col, row = dest
                self._cells_visited.add((col, row))
                result.teleported = True
                cell = int(self._cell_type[row, col])

            # ── confusion ──
            if cell == CELL_CONFUSION:
                if not self._confused_this_turn:
                    self._confused_count += 1
                    self._confused_this_turn = True
                is_confused = True                      # rest of this turn
                self._confused_turns_left = 2           # this turn remainder + next turn

            # ── death pit ──
            if cell == CELL_PIT:
                result.is_dead = True
                result.current_position = (col, row)
                result.is_confused = self._confused_this_turn or (self._confused_turns_left > 0)
                self._deaths += 1
                self._pos = self._start                 # respawn
                col, row  = self._start
                self._turn += 1
                # confusion persists through death
                if self._confused_turns_left > 0:
                    self._confused_turns_left -= 1
                return result

            # ── goal ──
            if cell == CELL_GOAL or (col, row) == self._goal:
                result.is_goal_reached = True
                result.current_position = (col, row)
                result.is_confused = self._confused_this_turn or (self._confused_turns_left > 0)
                self._goal_reached = True
                self._pos = (col, row)
                self._turn += 1
                if self._confused_turns_left > 0:
                    self._confused_turns_left -= 1
                return result

        # End of action list — episode continues
        self._pos = (col, row)
        result.current_position = (col, row)
        result.is_confused = self._confused_this_turn or (self._confused_turns_left > 0)

        self._turn += 1
        if self._confused_turns_left > 0:
            self._confused_turns_left -= 1

        return result

    def get_episode_stats(self) -> dict:
        return {
            "turns_taken":    self._turn,
            "deaths":         self._deaths,
            "confused":       self._confused_count,
            "cells_explored": len(self._cells_visited),
            "goal_reached":   self._goal_reached,
        }

    @property
    def start(self) -> Tuple[int, int]:
        return self._start

    @property
    def goal(self) -> Tuple[int, int]:
        return self._goal


# ─── agent base class (spec §6.1) ─────────────────────────────────────────────

class Agent:
    """Base class — students must implement plan_turn()."""

    def __init__(self) -> None:
        self.memory: dict = {}

    def plan_turn(self, last_result: Optional[TurnResult]) -> List[Action]:
        raise NotImplementedError

    def reset_episode(self) -> None:
        pass


# ─── smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    env = MazeEnvironment("training")
    start = env.reset()
    print(f"Start: {start},  Goal: {env.goal}")

    # Take a few manual steps to verify mechanics
    tests = [
        ([Action.MOVE_RIGHT],                    "move right from start"),
        ([Action.MOVE_UP],                       "move up"),
        ([Action.MOVE_LEFT, Action.MOVE_LEFT],   "two lefts"),
        ([Action.WAIT],                           "wait"),
    ]
    for actions, label in tests:
        result = env.step(actions)
        print(f"  [{label}] → {result}")

    print(f"\nStats: {env.get_episode_stats()}")
