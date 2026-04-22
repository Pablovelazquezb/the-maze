"""
Hybrid Q-Learning Agent — COSC 4368 Spring 2026
Phase 1: DFS-based wall mapping (state persists across episodes)
Phase 2: BFS optimal pathfinding + Q-Learning for hazard avoidance
"""

import random, pickle
from collections import defaultdict, deque
from environment import Action, Agent, TurnResult, ACTION_DELTA

ACTIONS = [Action.MOVE_UP, Action.MOVE_DOWN, Action.MOVE_LEFT, Action.MOVE_RIGHT]
DELTAS  = [(0,-1), (0,1), (-1,0), (1,0)]


class QLearningAgent(Agent):

    def __init__(self, alpha=0.15, gamma=0.95, epsilon=1.0,
                 epsilon_min=0.05, epsilon_decay=0.50):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma
        self.epsilon, self.epsilon_min, self.epsilon_decay = epsilon, epsilon_min, epsilon_decay

        # Q-table
        self.q_table = defaultdict(lambda: [0.0]*4)

        # Persistent map knowledge
        self.known_walls = set()
        self.known_open  = set()
        self.death_cells = set()
        self.teleport_map = {}
        self.confusion_cells = set()
        self.goal_pos = None
        self.start_pos = None

        # Persistent DFS frontier (survives episode resets)
        self.global_visited = set()

        # Episode state
        self.episodes_run = 0
        self._ep_reset()

    def _ep_reset(self):
        self.current_pos = None
        self.last_pos = None
        self.last_action_idx = None
        self.episode_path = []
        self.ep_visited = set()
        self.plan = []
        self.plan_step = 0
        self.replan_count = 0
        self.ep_deaths = 0

    def reset_episode(self):
        self._ep_reset()
        self.episodes_run += 1
        if self.episodes_run > 2:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ── Map helpers ───────────────────────────────────────────────────────────
    def _bfs(self, start, goal, avoid_deaths=True):
        """BFS shortest path on known-open edges. Ignores dynamic hazards so Q-learning can handle them."""
        queue = deque([(start, [])])
        visited = {start}
        while queue:
            pos, path = queue.popleft()
            if pos == goal:
                return path
            for ai in range(4):
                if (pos, ai) in self.known_walls:
                    continue
                dx, dy = DELTAS[ai]
                npos = (pos[0]+dx, pos[1]+dy)
                if npos in visited:
                    continue
                if avoid_deaths and npos in self.death_cells:
                    continue
                visited.add(npos)
                queue.append((npos, path + [ai]))
        if avoid_deaths:
            return self._bfs(start, goal, avoid_deaths=False)
        return None

    def _bfs_to_nearest_unexplored(self, start, avoid_deaths=True):
        """BFS to find the nearest cell with unexplored edges."""
        queue = deque([(start, [])])
        visited = {start}
        while queue:
            pos, path = queue.popleft()
            # Check if this cell has any unexplored direction
            for ai in range(4):
                if (pos, ai) in self.known_walls or (pos, ai) in self.known_open:
                    continue
                dx, dy = DELTAS[ai]
                npos = (pos[0]+dx, pos[1]+dy)
                if 0 <= npos[0] < 64 and 0 <= npos[1] < 64 and npos not in self.global_visited:
                    return path + [ai]
            # Expand through known edges (not walls)
            for ai in range(4):
                if (pos, ai) in self.known_walls:
                    continue
                dx, dy = DELTAS[ai]
                npos = (pos[0]+dx, pos[1]+dy)
                if npos not in visited and 0 <= npos[0] < 64 and 0 <= npos[1] < 64:
                    if avoid_deaths and npos in self.death_cells:
                        continue
                    visited.add(npos)
                    queue.append((npos, path + [ai]))
        if avoid_deaths:
            return self._bfs_to_nearest_unexplored(start, avoid_deaths=False)
        return None

    def _seed_q(self, path_actions, start):
        """Seed Q-values along optimal path."""
        pos = start
        n = len(path_actions)
        for i, ai in enumerate(path_actions):
            val = 80.0 * (n - i) / n + 20.0
            self.q_table[pos][ai] = max(self.q_table[pos][ai], val)
            dx, dy = DELTAS[ai]
            pos = (pos[0]+dx, pos[1]+dy)

    # ── Q-Learning update ─────────────────────────────────────────────────────
    def _update_q(self, state, ai, reward, next_state, terminal=False):
        old = self.q_table[state][ai]
        target = reward if terminal else reward + self.gamma * max(self.q_table[next_state])
        self.q_table[state][ai] = old + self.alpha * (target - old)

    # ── Main decision ─────────────────────────────────────────────────────────
    def plan_turn(self, last_result: TurnResult):
        if last_result is None:
            return [Action.WAIT]

        pos = last_result.current_position
        self.current_pos = pos
        self.ep_visited.add(pos)
        self.global_visited.add(pos)
        self.episode_path.append(pos)

        if self.start_pos is None:
            self.start_pos = pos

        # ── Learn from transition ─────────────────────────────────────────
        if self.last_pos is not None and self.last_action_idx is not None:
            lai = self.last_action_idx
            dx, dy = DELTAS[lai]
            expected = (self.last_pos[0]+dx, self.last_pos[1]+dy)

            if last_result.wall_hits > 0:
                self.known_walls.add((self.last_pos, lai))
                self._update_q(self.last_pos, lai, -10, pos)
                self.plan = []  # replan

            elif last_result.is_dead:
                self.death_cells.add(expected)
                self.ep_deaths += 1
                # Fire moves, so it's not a permanent wall! Just a negative reward for this state.
                self._update_q(self.last_pos, lai, -500, pos, terminal=True)
                self.plan = []

            elif last_result.is_goal_reached:
                self.goal_pos = pos
                self.known_open.add((self.last_pos, lai))
                self._update_q(self.last_pos, lai, 1000, pos, terminal=True)
                return [Action.WAIT]

            else:
                self.known_open.add((self.last_pos, lai))
                # Record reverse edge too
                rev = [1,0,3,2][lai]
                self.known_open.add((pos, rev))

                if last_result.teleported:
                    self.teleport_map[expected] = pos
                if last_result.is_confused:
                    self.confusion_cells.add(pos)

                reward = -1.0
                if pos not in self.global_visited:
                    reward += 3.0
                self._update_q(self.last_pos, lai, reward, pos)

        if last_result.is_goal_reached:
            self.goal_pos = pos
            return [Action.WAIT]

        # ── Choose action ─────────────────────────────────────────────────
        exploring = (self.goal_pos is None)

        if exploring:
            # Phase 1: Systematic exploration via BFS-to-frontier
            path = self._bfs_to_nearest_unexplored(pos)
            if path:
                ai = path[0]
            else:
                # Try random unexplored direction from current pos
                untried = [a for a in range(4)
                           if (pos,a) not in self.known_walls
                           and (pos,a) not in self.known_open]
                if untried:
                    ai = random.choice(untried)
                else:
                    # Fallback random
                    valid = [a for a in range(4) if (pos,a) not in self.known_walls]
                    ai = random.choice(valid) if valid else random.randint(0,3)
        else:
            # Phase 2: Navigate to goal via BFS + Q-Learning
            if not self.plan or self.plan_step >= len(self.plan):
                path = self._bfs(pos, self.goal_pos)
                if path:
                    self.plan = path
                    self.plan_step = 0
                    if self.episodes_run <= 5:
                        self._seed_q(path, pos)
                else:
                    self.plan = []

            if random.random() < self.epsilon:
                # Explore: try untried directions or random
                valid = [a for a in range(4) if (pos,a) not in self.known_walls]
                ai = random.choice(valid) if valid else random.randint(0,3)
            elif self.plan and self.plan_step < len(self.plan):
                # Follow BFS plan unless Q-value is terrible (meaning hazard is here)
                planned_ai = self.plan[self.plan_step]
                qvals = self.q_table[pos]
                if qvals[planned_ai] < -50:
                    # Hazard blocking the planned path! Re-evaluate with Q-values
                    ai = qvals.index(max(qvals))
                    self.plan = [] # force replan next step
                else:
                    ai = planned_ai
                    self.plan_step += 1
            else:
                qvals = list(self.q_table[pos])
                for a in range(4):
                    if (pos, a) in self.known_walls:
                        qvals[a] = -99999
                ai = qvals.index(max(qvals))

        self.last_pos = pos
        self.last_action_idx = ai
        return [ACTIONS[ai]]

    # ── Save / Load ───────────────────────────────────────────────────────────
    def save(self, filename):
        data = {
            'q_table': dict(self.q_table),
            'known_walls': self.known_walls,
            'known_open': self.known_open,
            'death_cells': self.death_cells,
            'teleport_map': self.teleport_map,
            'confusion_cells': self.confusion_cells,
            'goal_pos': self.goal_pos,
            'start_pos': self.start_pos,
            'global_visited': self.global_visited,
            'episodes_run': self.episodes_run,
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    def load(self, filename):
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            self.q_table = defaultdict(lambda: [0.0]*4, data['q_table'])
            self.known_walls = data['known_walls']
            self.known_open = data['known_open']
            self.death_cells = data['death_cells']
            self.teleport_map = data.get('teleport_map', {})
            self.confusion_cells = data.get('confusion_cells', set())
            self.goal_pos = data['goal_pos']
            self.start_pos = data['start_pos']
            self.global_visited = data.get('global_visited', set())
            self.episodes_run = data.get('episodes_run', 0)
            return True
        except Exception as e:
            print(f"Load failed: {e}")
            return False
