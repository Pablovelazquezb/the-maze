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

import heapq
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


# ─── SmartAgent ──────────────────────────────────────────────────────────────

class SmartAgent(NaiveAgent):
    """
    Improvement over NaiveAgent:
      1. A* replaces BFS for navigation and frontier selection.
         A* biases exploration toward the goal (or the diagonal-opposite corner
         as heuristic when goal is unknown), finding the goal earlier in ep 1.
      2. Monte Carlo Q-learning accumulates (state, action) value estimates
         across episodes.  From episode 2 onward the Q-table biases which
         unexplored direction is probed first, cutting turns further.
    """

    _ALPHA        = 0.20
    _GAMMA        = 0.92
    _GOAL_REWARD  = 5_000.0
    _DEATH_PENALTY = -300.0
    _STEP_COST    = -1.0

    def __init__(self, start: Tuple[int, int]) -> None:
        super().__init__(start)
        self._q:        Dict[Tuple[int, int], Dict[Action, float]] = {}
        self._history:  List[Tuple[Tuple[int, int], Action]] = []
        self._ep_deaths: int = 0

    # ── episode bookkeeping ───────────────────────────────────────────────────

    def reset_episode(self) -> None:
        # Detect goal reach BEFORE resetting position (pos == goal at ep end).
        ep_goal = (
            self.memory["goal"] is not None
            and self._pos == self.memory["goal"]
        )
        self._flush_q(ep_goal)
        self._history   = []
        self._ep_deaths = 0
        super().reset_episode()

    def _update_model(self, result: Optional[TurnResult]) -> None:
        if result is not None and result.is_dead:
            self._ep_deaths += 1
        super()._update_model(result)

    # ── Q-learning ────────────────────────────────────────────────────────────

    def _qv(self, pos: Tuple[int, int], action: Action) -> float:
        return self._q.get(pos, {}).get(action, 0.0)

    def _record_move(self, pos: Tuple[int, int], action: Action) -> None:
        self._history.append((pos, action))

    def _flush_q(self, goal_reached: bool) -> None:
        """Monte Carlo backward update through the episode history."""
        if not self._history:
            return
        G = self._GOAL_REWARD if goal_reached else 0.0
        G += self._DEATH_PENALTY * self._ep_deaths
        for state, action in reversed(self._history):
            G = self._STEP_COST + self._GAMMA * G
            entry = self._q.setdefault(state, {})
            entry[action] = (
                entry.get(action, 0.0)
                + self._ALPHA * (G - entry.get(action, 0.0))
            )

    # ── heuristic & A* ───────────────────────────────────────────────────────

    def _target(self) -> Optional[Tuple[int, int]]:
        """Goal if known; else None (use BFS for exploration)."""
        return self.memory["goal"]

    @staticmethod
    def _mh(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _astar_path(
        self, src: Tuple[int, int], dst: Tuple[int, int]
    ) -> Optional[List[Tuple[int, int]]]:
        if src == dst:
            return [src]
        open_set: list = [(self._mh(src, dst), 0, src)]
        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {src: None}
        g_cost:    Dict[Tuple[int, int], int] = {src: 0}
        while open_set:
            _, cost, cur = heapq.heappop(open_set)
            if cur == dst:
                path, node = [], cur
                while node is not None:
                    path.append(node)
                    node = came_from[node]
                path.reverse()
                return path
            if cost > g_cost.get(cur, 10**9):
                continue
            for nxt in self._passable_neighbours(cur):
                if not self._is_safe(nxt):
                    continue
                ng = cost + 1
                if ng < g_cost.get(nxt, 10**9):
                    g_cost[nxt] = ng
                    came_from[nxt] = cur
                    heapq.heappush(open_set, (ng + self._mh(nxt, dst), ng, nxt))
        return None

    def _astar_to_frontier(self) -> Optional[List[Tuple[int, int]]]:
        """A* toward target; falls back to BFS when goal is unknown."""
        tgt = self._target()
        if tgt is None:
            return self._bfs_to_frontier()
        src = self._pos
        open_set: list = [(self._mh(src, tgt), 0, src)]
        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {src: None}
        g_cost:    Dict[Tuple[int, int], int] = {src: 0}
        while open_set:
            _, cost, cur = heapq.heappop(open_set)
            if cur != src and self._unexplored_dirs(cur):
                path, node = [], cur
                while node is not None:
                    path.append(node)
                    node = came_from[node]
                path.reverse()
                return path
            if cost > g_cost.get(cur, 10**9):
                continue
            for nxt in self._passable_neighbours(cur):
                if not self._is_safe(nxt):
                    continue
                ng = cost + 1
                if ng < g_cost.get(nxt, 10**9):
                    g_cost[nxt] = ng
                    came_from[nxt] = cur
                    heapq.heappush(open_set, (ng + self._mh(nxt, tgt), ng, nxt))
        return None

    # ── probe direction selection ─────────────────────────────────────────────

    def _best_probe(
        self, pos: Tuple[int, int]
    ) -> Optional[Tuple[Action, int, int]]:
        """
        Pick the unexplored direction maximising Q-value.
        When goal is known, adds proximity-to-goal bias.
        """
        candidates = self._unexplored_dirs(pos)
        if not candidates:
            return None
        tgt = self._target()
        if tgt is None:
            # No goal known yet — rank by Q-value only, fall back to first
            return max(candidates, key=lambda item: self._qv(pos, item[0]))
        norm = N * 2.0
        def rank(item: Tuple[Action, int, int]) -> float:
            action, nc, nr = item
            return self._qv(pos, action) - self._mh((nc, nr), tgt) / norm
        return max(candidates, key=rank)

    # ── plan_turn override ────────────────────────────────────────────────────

    def plan_turn(self, last_result: Optional[TurnResult]) -> List[Action]:
        self._update_model(last_result)

        confused = self._confused_turns_left > 0

        def _emit(actions: List[Action]) -> List[Action]:
            return [_INVERT[a] for a in actions] if confused else actions

        # ── 1. continue navigation plan (identical to NaiveAgent) ─────────────
        if self._nav_positions:
            nxt = self._nav_positions[0]
            if (
                self._edge(self._pos, nxt) in self.memory["open_edges"]
                and self._is_safe(nxt)
            ):
                if not self._is_probe_safe(nxt):
                    self._probe_target = nxt
                    self._nav_positions = self._nav_positions[1:]
                    return _emit([self._pos_to_action(self._pos, nxt)])

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

        # ── 2. head to goal (A*) ───────────────────────────────────────────────
        goal = self.memory["goal"]
        if goal is not None and self._pos != goal:
            path = self._astar_path(self._pos, goal)
            if path and len(path) > 1:
                self._nav_positions = path[1:]
                return self.plan_turn(None)

        # ── 3. probe best unexplored direction (Q + heuristic) ────────────────
        best = self._best_probe(self._pos)
        if best:
            action, nc, nr = best
            self._probe_target = (nc, nr)
            self._record_move(self._pos, action)
            return _emit([action])

        # ── 4. navigate to frontier (A*-biased toward target) ─────────────────
        path = self._astar_to_frontier()
        if path and len(path) > 1:
            self._nav_positions = path[1:]
            return self.plan_turn(None)

        # ── 5. escape via intentional pit death ───────────────────────────────
        path = self._bfs_to_escape_pit()
        if path and len(path) > 1:
            pit_cell = path[-1]
            nav_to_adj = path[1:-1]
            if nav_to_adj:
                self._nav_positions = nav_to_adj
                return self.plan_turn(None)
            self._probe_target = pit_cell
            return _emit([self._pos_to_action(self._pos, pit_cell)])

        # ── 6. nothing reachable ──────────────────────────────────────────────
        return [Action.WAIT]


# ─── utility ─────────────────────────────────────────────────────────────────

def _adjacent(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


# ─── episode runner ───────────────────────────────────────────────────────────

def run_episodes(
    maze_id:   str  = "training",
    n:         int  = 5,
    max_turns: int  = 10_000,
    verbose:   bool = False,
    agent_cls        = None,
) -> List[dict]:
    """Run n episodes with persistent memory. Returns per-episode stats."""
    from environment import MazeEnvironment

    if agent_cls is None:
        agent_cls = NaiveAgent

    env   = MazeEnvironment(maze_id)
    start = env.reset()
    agent = agent_cls(start)

    results = []
    for ep in range(1, n + 1):
        env.reset()
        agent.reset_episode()
        last_result: Optional[TurnResult] = None

        for turn in range(max_turns):
            actions     = agent.plan_turn(last_result)
            last_result = env.step(actions)

            if verbose and turn % 500 == 0:
                s = env.get_episode_stats()
                print(
                    f"  ep={ep} turn={turn:5d}  pos={last_result.current_position}"
                    f"  explored={s['cells_explored']:4d}/4096"
                    f"  deaths={s['deaths']}  confused={s['confused']}"
                )

            if last_result.is_goal_reached:
                if verbose:
                    print(f"  *** GOAL reached on turn {turn + 1}!")
                break

        results.append(env.get_episode_stats())

    return results


# ─── smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import statistics

    for label, cls in [("NaiveAgent", NaiveAgent), ("SmartAgent", SmartAgent)]:
        print(f"\n{'='*50}")
        print(f"  {label} — training maze, 5 episodes")
        print(f"{'='*50}")
        results = run_episodes("training", n=5, verbose=True, agent_cls=cls)
        successes = [r for r in results if r["goal_reached"]]
        turns  = [r["turns_taken"] for r in successes]
        deaths = [r["deaths"]      for r in successes]
        total_d = sum(r["deaths"]      for r in results)
        total_t = sum(r["turns_taken"] for r in results)
        print(f"\n  success={len(successes)}/5"
              f"  avg_turns={statistics.mean(turns):.0f}"
              f"  avg_deaths={statistics.mean(deaths):.2f}"
              f"  death_rate={total_d/total_t:.5f}")
