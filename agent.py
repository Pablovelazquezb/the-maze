"""
NaiveAgent — systematic BFS frontier explorer.

Exploration strategy:
  PROBE:    Send exactly ONE action to test an unexplored direction (or to enter
            an unknown cell for the first time).  Record edge type and cell type
            from the result.
  NAVIGATE: When current cell is fully explored and the next hop is a confirmed-
            safe cell (in memory["probed"]), batch up to 5 steps per turn along
            a path to the nearest frontier.  Unknown cells are always approached
            one step at a time (treated as probes) so confusion cells are
            identified before being bulk-traversed.
  GOAL:     Once the goal is known, navigate to it using the same mechanism.

World model (self.memory):
  open_edges   : set[frozenset] — confirmed passable cell boundaries
  closed_edges : set[frozenset] — confirmed walls
  probed       : set[(col,row)] — cells entered via probe, confirmed non-confusion
  hazards      : dict (col,row) → cell_type  (PIT, CONFUSION, GOAL, TELEPORT)
  teleport_map : dict (col,row) → (col,row)
  goal         : (col,row) or None
  start        : (col,row)

Critical invariant: closed edges are ONLY recorded when BOTH environment AND
agent report no confusion, to prevent confusion-inversion from creating false walls.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

from environment import Action, Agent, TurnResult
from maze_parser import CELL_CONFUSION, CELL_GOAL, CELL_PIT, CELL_TELEPORT, GRID_SIZE

N = GRID_SIZE

_DIRS: List[Tuple[Action, int, int]] = [
    (Action.MOVE_UP,    0, -1),
    (Action.MOVE_DOWN,  0,  1),
    (Action.MOVE_LEFT, -1,  0),
    (Action.MOVE_RIGHT, 1,  0),
]

_INVERT: Dict[Action, Action] = {
    Action.MOVE_UP:    Action.MOVE_DOWN,
    Action.MOVE_DOWN:  Action.MOVE_UP,
    Action.MOVE_LEFT:  Action.MOVE_RIGHT,
    Action.MOVE_RIGHT: Action.MOVE_LEFT,
    Action.WAIT:       Action.WAIT,
}


class NaiveAgent(Agent):

    def __init__(self, start: Tuple[int, int]) -> None:
        super().__init__()
        self._start = start
        self._pos   = start

        # Set when sending a single probe; None during multi-step navigation.
        self._probe_target: Optional[Tuple[int, int]] = None

        # Planned positions left to visit on the current navigation plan.
        self._nav_positions: List[Tuple[int, int]] = []

        # Confusion state mirrored from environment feedback.
        self._confused_turns_left: int = 0

        self.memory: dict = {
            "open_edges":   set(),
            "closed_edges": set(),
            "probed":       {start},   # cells confirmed non-confusion
            "hazards":      {},
            "teleport_map": {},
            "goal":         None,
            "start":        start,
        }

    def reset_episode(self) -> None:
        self._pos = self._start
        self._probe_target = None
        self._nav_positions = []
        self._confused_turns_left = 0
        # World model persists across episodes.

    # ── world model helpers ───────────────────────────────────────────────────

    def _edge(self, a: Tuple[int, int], b: Tuple[int, int]) -> frozenset:
        return frozenset({a, b})

    def _record_open(self, a: Tuple[int, int], b: Tuple[int, int]) -> None:
        e = self._edge(a, b)
        self.memory["open_edges"].add(e)
        self.memory["closed_edges"].discard(e)

    def _record_closed(self, a: Tuple[int, int], b: Tuple[int, int]) -> None:
        e = self._edge(a, b)
        self.memory["closed_edges"].add(e)
        self.memory["open_edges"].discard(e)

    def _passable_neighbours(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        result = []
        col, row = pos
        for _, dc, dr in _DIRS:
            nc, nr = col + dc, row + dr
            if 0 <= nc < N and 0 <= nr < N:
                if self._edge(pos, (nc, nr)) in self.memory["open_edges"]:
                    result.append((nc, nr))
        return result

    def _unexplored_dirs(
        self, pos: Tuple[int, int]
    ) -> List[Tuple[Action, int, int]]:
        col, row = pos
        result = []
        for action, dc, dr in _DIRS:
            nc, nr = col + dc, row + dr
            if not (0 <= nc < N and 0 <= nr < N):
                continue
            e = self._edge(pos, (nc, nr))
            if e not in self.memory["open_edges"] and e not in self.memory["closed_edges"]:
                result.append((action, nc, nr))
        return result

    def _is_safe(self, pos: Tuple[int, int]) -> bool:
        """Exclude pits and teleporters from BFS navigation paths."""
        h = self.memory["hazards"].get(pos, 0)
        return h != CELL_PIT and h != CELL_TELEPORT

    def _is_probe_safe(self, pos: Tuple[int, int]) -> bool:
        """A confirmed-safe cell we can include in a navigation batch."""
        return (
            self.memory["hazards"].get(pos, 0) != CELL_PIT
            and pos in self.memory["probed"]
        )

    # ── BFS ───────────────────────────────────────────────────────────────────

    def _bfs_path(
        self, src: Tuple[int, int], dst: Tuple[int, int]
    ) -> Optional[List[Tuple[int, int]]]:
        if src == dst:
            return [src]
        parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {src: None}
        q: deque = deque([src])
        while q:
            cur = q.popleft()
            for nxt in self._passable_neighbours(cur):
                if nxt in parent or not self._is_safe(nxt):
                    continue
                parent[nxt] = cur
                if nxt == dst:
                    path, node = [], nxt
                    while node is not None:
                        path.append(node)
                        node = parent[node]
                    path.reverse()
                    return path
                q.append(nxt)
        return None

    def _bfs_to_frontier(self) -> Optional[List[Tuple[int, int]]]:
        """Path from current pos to nearest cell with ≥1 unexplored edge."""
        src = self._pos
        parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {src: None}
        q: deque = deque([src])
        while q:
            cur = q.popleft()
            if cur != src and self._unexplored_dirs(cur):
                path, node = [], cur
                while node is not None:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                return path
            for nxt in self._passable_neighbours(cur):
                if nxt in parent or not self._is_safe(nxt):
                    continue
                parent[nxt] = cur
                q.append(nxt)
        return None

    def _bfs_to_escape_pit(self) -> Optional[List[Tuple[int, int]]]:
        """Path from current pos to nearest reachable pit cell for intentional respawn."""
        src = self._pos
        hazards = self.memory["hazards"]
        parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {src: None}
        q: deque = deque([src])
        while q:
            cur = q.popleft()
            for _, dc, dr in _DIRS:
                nc, nr = cur[0] + dc, cur[1] + dr
                nxt = (nc, nr)
                if not (0 <= nc < N and 0 <= nr < N):
                    continue
                if self._edge(cur, nxt) not in self.memory["open_edges"]:
                    continue
                if hazards.get(nxt) == CELL_PIT:
                    path, node = [], cur
                    while node is not None:
                        path.append(node)
                        node = parent[node]
                    path.reverse()
                    return path + [nxt]
                if nxt in parent or not self._is_safe(nxt):
                    continue
                parent[nxt] = cur
                q.append(nxt)
        return None

    @staticmethod
    def _pos_to_action(src: Tuple[int, int], dst: Tuple[int, int]) -> Action:
        dc, dr = dst[0] - src[0], dst[1] - src[1]
        for a, adc, adr in _DIRS:
            if adc == dc and adr == dr:
                return a
        raise ValueError(f"Non-adjacent: {src} → {dst}")

    # ── world model update ────────────────────────────────────────────────────

    def _update_model(self, result: Optional[TurnResult]) -> None:
        if result is None:
            return

        prev  = self._pos
        new   = result.current_position
        probe = self._probe_target
        self._probe_target = None

        # Was the agent already confused BEFORE this turn's actions?
        was_pre_confused = self._confused_turns_left > 0

        # Update confusion counter to mirror the environment exactly.
        # env._confused_turns_left is decremented at the end of each step, so:
        #   - Fresh trigger (was_pre_confused=False, is_confused=True):
        #     env was 0→2→1 (decremented). Next turn: env is confused (1>0).
        #     Set agent to 1 — confused exactly ONE more turn.
        #   - Carryover last turn (was_pre_confused=True, is_confused=True):
        #     env was 1→decremented→0. Next turn: env NOT confused.
        #     Set agent to 0 — done.
        #   - Not confused: decrement normally.
        if result.is_confused:
            self._confused_turns_left = 0 if was_pre_confused else 1
        elif self._confused_turns_left > 0:
            self._confused_turns_left -= 1

        # ── death pit ──────────────────────────────────────────────────────────
        if result.is_dead:
            self.memory["hazards"][new] = CELL_PIT
            if _adjacent(prev, new):
                self._record_open(prev, new)
            self._pos = self._start
            self._nav_positions = []
            return

        # ── goal ───────────────────────────────────────────────────────────────
        if result.is_goal_reached:
            self.memory["hazards"][new] = CELL_GOAL
            self.memory["goal"] = new
            if _adjacent(prev, new):
                self._record_open(prev, new)
            self._pos = new
            return

        # ── teleport ───────────────────────────────────────────────────────────
        if result.teleported:
            # The teleporter is the cell we probed toward, not prev.
            # prev is the safe cell the agent stepped FROM.
            teleporter = probe if probe is not None else prev
            if probe is not None:
                self._record_open(prev, probe)   # passage to teleporter is passable
            self.memory["teleport_map"][teleporter] = new
            self.memory["hazards"][teleporter] = CELL_TELEPORT
            self._pos = new
            self._nav_positions = []
            return

        # ── probe (single-step) ────────────────────────────────────────────────
        if probe is not None:
            if prev != new:
                self._record_open(prev, new)
                self._pos = new
                if result.is_confused and not was_pre_confused:
                    # Freshly triggered confusion — the new cell caused it.
                    self.memory["hazards"][new] = CELL_CONFUSION
                else:
                    # Cell is confirmed safe (confusion is carryover, not from here).
                    self.memory["probed"].add(new)
            else:
                # Wall hit — only record closed when BOTH sides are not confused,
                # to avoid false walls from confusion-inversion.
                if not result.is_confused and self._confused_turns_left == 0:
                    self._record_closed(prev, probe)
            return

        # ── navigation batch ───────────────────────────────────────────────────
        # All traversed edges are already in open_edges.  Just update position.
        # Do NOT call _record_open (would add bogus long-range edge) or mark
        # confusion cells (can't tell which batch step triggered it).
        self._pos = new
        if result.is_confused and not was_pre_confused:
            # Confusion triggered mid-batch.  Clear plan so we re-probe from here.
            self._nav_positions = []

    # ── planning ──────────────────────────────────────────────────────────────

    def plan_turn(self, last_result: Optional[TurnResult]) -> List[Action]:
        self._update_model(last_result)

        confused = self._confused_turns_left > 0

        def _emit(actions: List[Action]) -> List[Action]:
            return [_INVERT[a] for a in actions] if confused else actions

        # ── 1. continue navigation plan ────────────────────────────────────────
        if self._nav_positions:
            nxt = self._nav_positions[0]
            if (
                self._edge(self._pos, nxt) in self.memory["open_edges"]
                and self._is_safe(nxt)
            ):
                if not self._is_probe_safe(nxt):
                    # Unknown cell — approach as a single-step probe.
                    self._probe_target = nxt
                    self._nav_positions = self._nav_positions[1:]
                    return _emit([self._pos_to_action(self._pos, nxt)])

                # Confirmed-safe cell — batch up to 5 consecutive known steps.
                batch: List[Tuple[int, int]] = []
                cur = self._pos
                for p in self._nav_positions:
                    if (
                        self._edge(cur, p) in self.memory["open_edges"]
                        and self._is_probe_safe(p)
                        and len(batch) < 5
                    ):
                        batch.append(p)
                        cur = p
                    else:
                        break
                self._nav_positions = self._nav_positions[len(batch):]
                actions = [
                    self._pos_to_action(batch[i - 1] if i > 0 else self._pos, batch[i])
                    for i in range(len(batch))
                ]
                return _emit(actions)
            else:
                self._nav_positions = []

        # ── 2. head to goal if known and reachable ─────────────────────────────
        goal = self.memory["goal"]
        if goal is not None and self._pos != goal:
            path = self._bfs_path(self._pos, goal)
            if path and len(path) > 1:
                self._nav_positions = path[1:]
                return self.plan_turn(None)

        # ── 3. probe an unexplored direction at current cell ───────────────────
        unexplored = self._unexplored_dirs(self._pos)
        if unexplored:
            action, nc, nr = unexplored[0]
            self._probe_target = (nc, nr)
            return _emit([action])

        # ── 4. navigate to nearest frontier ───────────────────────────────────
        path = self._bfs_to_frontier()
        if path and len(path) > 1:
            self._nav_positions = path[1:]
            return self.plan_turn(None)

        # ── 5. escape via intentional pit death (teleport isolation recovery) ────
        path = self._bfs_to_escape_pit()
        if path and len(path) > 1:
            pit_cell = path[-1]
            nav_to_adj = path[1:-1]  # safe cells between here and the pit
            if nav_to_adj:
                self._nav_positions = nav_to_adj
                return self.plan_turn(None)
            # Already adjacent — probe-step into the pit to force respawn.
            self._probe_target = pit_cell
            return _emit([self._pos_to_action(self._pos, pit_cell)])

        # ── 6. nothing reachable ───────────────────────────────────────────────
        return [Action.WAIT]


# ─── utility ─────────────────────────────────────────────────────────────────

def _adjacent(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


# ─── episode runner ───────────────────────────────────────────────────────────

def run_episode(
    maze_id: str = "training",
    max_turns: int = 10_000,
    verbose: bool = False,
) -> dict:
    from environment import MazeEnvironment

    env   = MazeEnvironment(maze_id)
    start = env.reset()
    agent = NaiveAgent(start)

    last_result: Optional[TurnResult] = None
    for turn in range(max_turns):
        actions = agent.plan_turn(last_result)
        last_result = env.step(actions)

        if verbose and turn % 500 == 0:
            s = env.get_episode_stats()
            print(
                f"  turn={turn:5d}  pos={last_result.current_position}"
                f"  explored={s['cells_explored']:4d}/4096"
                f"  deaths={s['deaths']}"
                f"  confused={s['confused']}"
                f"  probed={len(agent.memory['probed'])}"
            )

        if last_result.is_goal_reached:
            if verbose:
                print(f"  *** GOAL reached on turn {turn + 1}!")
            break

    return env.get_episode_stats()


# ─── smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running NaiveAgent on training maze...")
    stats = run_episode("training", max_turns=10_000, verbose=True)
    print(f"\nEpisode stats: {stats}")
