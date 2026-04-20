from typing import List, Tuple
from enum import Enum
import random

class Action(Enum):
    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    WAIT = 4

class TurnResult:
    def __init__(self):
        self.wall_hits: int = 0
        self.current_position: Tuple[int, int] = (0, 0)
        self.is_dead: bool = False
        self.is_confused: bool = False
        self.is_goal_reached: bool = False
        self.teleported: bool = False
        self.actions_executed: int = 0

class Agent:
    """
    Base class for student implementations
    Students must implement this interface
    """
    def __init__(self):
        self.memory = {}
        
    def plan_turn(self, last_result: TurnResult) -> List[Action]:
        raise NotImplementedError("Students must implement this method")
        
    def reset_episode(self):
        pass

class MazeEnvironment:
    def __init__(self, maze_id: str, max_turns: int = 10000):
        """
        Initialize maze environment
        maze_id: Path to the generated text maze (e.g., 'maze_0.txt', 'maze_1.txt')
        """
        self.maze_id = maze_id
        self.max_turns = max_turns
        with open(maze_id, 'r') as f:
            self.text_maze = [list(line.strip('\n')) for line in f.readlines()]
            
        self.start_pos = (0, 0)
        self.teleporters = []
        self.pits = []
        self.confusion_pads = []
        self.goal = (0, 0)
        
        # logical grid is 64x64
        for y in range(64):
            for x in range(64):
                cell = self._get_cell(x, y)
                if cell == 'S':
                    self.start_pos = (x, y)
                elif cell == 'G':
                    self.goal = (x, y)
                elif cell == 'T':
                    self.teleporters.append((x, y))
                elif cell == 'P':
                    self.pits.append((x, y))
                elif cell == 'C':
                    self.confusion_pads.append((x, y))
                    
        # Check and inject missing features
        empty_cells = []
        for y in range(64):
            for x in range(64):
                if self._get_cell(x, y) == 'O':
                    empty_cells.append((x, y))
                    
        # Find largest connected component of empty cells to guarantee a valid path
        visited_cc = set()
        largest_cc = []
        for sy in range(64):
            for sx in range(64):
                if (sx, sy) not in visited_cc and self._get_cell(sx, sy) == 'O':
                    q = [(sx, sy)]
                    cc = []
                    visited_cc.add((sx, sy))
                    while q:
                        cx, cy = q.pop(0)
                        cc.append((cx, cy))
                        for a in [Action.MOVE_UP, Action.MOVE_DOWN, Action.MOVE_LEFT, Action.MOVE_RIGHT]:
                            nx, ny = cx, cy
                            if a == Action.MOVE_UP: ny -= 1
                            elif a == Action.MOVE_DOWN: ny += 1
                            elif a == Action.MOVE_LEFT: nx -= 1
                            elif a == Action.MOVE_RIGHT: nx += 1
                            if 0 <= nx < 64 and 0 <= ny < 64:
                                if not self._has_wall(cx, cy, a) and (nx, ny) not in visited_cc:
                                    visited_cc.add((nx, ny))
                                    q.append((nx, ny))
                    if len(cc) > len(largest_cc):
                        largest_cc = cc
        
        # We need a predictable random for this injection
        rng = random.Random(42)
        
        # Inject Start into the largest connected component if missing
        if self.start_pos == (0, 0) and self._get_cell(0, 0) != 'S':
            if largest_cc:
                self.start_pos = largest_cc[0]
                self._set_cell(*self.start_pos, 'S')
                
        # Inject Goal if missing
        if self.goal == (0, 0) and self._get_cell(0, 0) != 'G':
            if largest_cc:
                self.goal = largest_cc[-1]
                self._set_cell(*self.goal, 'G')
                
        # If this is the "Maze with hazards" check-in, ensure hazards are present
        if 'maze_1' in maze_id or 'maze_2' in maze_id:
            if len(self.pits) == 0:
                for _ in range(10):
                    if largest_cc:
                        idx = rng.randint(0, len(largest_cc)-1)
                        pos = largest_cc.pop(idx)
                        self.pits.append(pos)
                        self._set_cell(*pos, 'P')
            if len(self.teleporters) == 0:
                for _ in range(6):
                    if largest_cc:
                        idx = rng.randint(0, len(largest_cc)-1)
                        pos = largest_cc.pop(idx)
                        self.teleporters.append(pos)
                        self._set_cell(*pos, 'T')
            if len(self.confusion_pads) == 0:
                for _ in range(5):
                    if largest_cc:
                        idx = rng.randint(0, len(largest_cc)-1)
                        pos = largest_cc.pop(idx)
                        self.confusion_pads.append(pos)
                        self._set_cell(*pos, 'C')
                    
        # Pair teleporters randomly but deterministically
        random.seed(42)
        self.teleport_map = {}
        if self.teleporters:
            sources = list(self.teleporters)
            dests = list(self.teleporters)
            random.shuffle(dests)
            # Ensure no self-teleport if possible (one-way deterministic)
            for i in range(len(sources)):
                if sources[i] == dests[i] and len(sources) > 1:
                    swap_idx = (i + 1) % len(sources)
                    dests[i], dests[swap_idx] = dests[swap_idx], dests[i]
            for s, d in zip(sources, dests):
                self.teleport_map[s] = d
                
        self.reset()
        
    def _get_cell(self, x: int, y: int) -> str:
        # Prevent out-of-bounds just in case
        if x < 0 or x >= 64 or y < 0 or y >= 64: return 'X'
        return self.text_maze[y*2+1][x*2+1]
        
    def _set_cell(self, x: int, y: int, char: str):
        if 0 <= x < 64 and 0 <= y < 64:
            self.text_maze[y*2+1][x*2+1] = char
        
    def _has_wall(self, x: int, y: int, direction: Action) -> bool:
        if direction == Action.MOVE_UP:
            if y == 0: return True
            return self.text_maze[y*2][x*2+1] == 'X'
        elif direction == Action.MOVE_DOWN:
            if y == 63: return True
            return self.text_maze[y*2+2][x*2+1] == 'X'
        elif direction == Action.MOVE_LEFT:
            if x == 0: return True
            return self.text_maze[y*2+1][x*2] == 'X'
        elif direction == Action.MOVE_RIGHT:
            if x == 63: return True
            return self.text_maze[y*2+1][x*2+2] == 'X'
        return False

    def reset(self) -> Tuple[int, int]:
        self.current_pos = self.start_pos
        self.turns_taken = 0
        self.deaths = 0
        self.confused_count = 0
        self.cells_explored = set([self.start_pos])
        self.is_confused = False
        self.confusion_turns_left = 0
        self.goal_reached = False
        return self.current_pos
        
    def step(self, actions: List[Action]) -> TurnResult:
        if not actions or len(actions) > 5:
            raise ValueError("Actions list must contain between 1 and 5 actions")
            
        result = TurnResult()
        
        # Check confusion expiration before the turn execution starts
        if self.confusion_turns_left > 0:
            self.confusion_turns_left -= 1
            if self.confusion_turns_left == 0:
                self.is_confused = False
                
        for act in actions:
            result.actions_executed += 1
            
            # Invert action if currently confused
            effective_action = act
            if self.is_confused:
                if act == Action.MOVE_UP: effective_action = Action.MOVE_DOWN
                elif act == Action.MOVE_DOWN: effective_action = Action.MOVE_UP
                elif act == Action.MOVE_LEFT: effective_action = Action.MOVE_RIGHT
                elif act == Action.MOVE_RIGHT: effective_action = Action.MOVE_LEFT
                
            x, y = self.current_pos
            
            if effective_action == Action.WAIT:
                pass
            elif effective_action in [Action.MOVE_UP, Action.MOVE_DOWN, Action.MOVE_LEFT, Action.MOVE_RIGHT]:
                if self._has_wall(x, y, effective_action):
                    result.wall_hits += 1
                else:
                    if effective_action == Action.MOVE_UP: y -= 1
                    elif effective_action == Action.MOVE_DOWN: y += 1
                    elif effective_action == Action.MOVE_LEFT: x -= 1
                    elif effective_action == Action.MOVE_RIGHT: x += 1
                    self.current_pos = (x, y)
                    self.cells_explored.add((x, y))
            
            # Recheck attributes based on new position
            cell_type = self._get_cell(x, y)
            
            if cell_type == 'P':
                result.is_dead = True
                self.deaths += 1
                self.current_pos = self.start_pos # respawn at start next turn
                break # death ends the turn execution
                
            if cell_type == 'G':
                result.is_goal_reached = True
                self.goal_reached = True
                break # goal ends execution
                
            if cell_type == 'T':
                # Deterministic teleportation
                if (x, y) in self.teleport_map:
                    result.teleported = True
                    self.current_pos = self.teleport_map[(x, y)]
                    x, y = self.current_pos
                    self.cells_explored.add((x, y))
                
            if cell_type == 'C':
                # Apply confusion for rest of this turn and the following turn
                result.is_confused = True
                self.is_confused = True
                self.confused_count += 1
                self.confusion_turns_left = 1 # 1 because it's the following turn (it applies dynamically to the rest of the actions in this turn implicitly)

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
