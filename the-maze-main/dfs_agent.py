from environment import Action, TurnResult, Agent
from typing import List

class DFSAgent(Agent):
    def __init__(self):
        super().__init__()
        self.path = []  # Stack of (x,y) to represent current path from start
        self.visited = set()
        self.blocked_edges = set()
        self.current_pos = None
        self.backtracking = False
        self.untried_actions = {} 
        self.last_action = None
        
    def reset_episode(self):
        self.path = []
        self.visited = set()
        self.blocked_edges = set()
        self.current_pos = None
        self.untried_actions = {}
        self.last_action = None
        self.backtracking = False

    def plan_turn(self, last_result: TurnResult) -> List[Action]:
        if last_result is None:
            # First turn: do a WAIT to just observe the starting position
            return [Action.WAIT]
            
        pos = last_result.current_position
        self.current_pos = pos
        
        # If we hit a wall last turn
        if last_result.wall_hits > 0 and self.last_action is not None and not self.backtracking:
            # Record wall location as a blocked edge
            self.blocked_edges.add((pos, self.last_action))
            hx, hy = pos
            if self.last_action == Action.MOVE_UP: hy -= 1; rev = Action.MOVE_DOWN
            elif self.last_action == Action.MOVE_DOWN: hy += 1; rev = Action.MOVE_UP
            elif self.last_action == Action.MOVE_LEFT: hx -= 1; rev = Action.MOVE_RIGHT
            elif self.last_action == Action.MOVE_RIGHT: hx += 1; rev = Action.MOVE_LEFT
            self.blocked_edges.add(((hx, hy), rev))
            
        elif self.last_action is not None and not self.backtracking:
            # Successfully entered a new cell
            if pos not in self.visited:
                self.visited.add(pos)
                if len(self.path) == 0 or self.path[-1] != pos:
                    self.path.append(pos)
                    
        if pos not in self.visited:
            self.visited.add(pos)
            self.path.append(pos)
            
        if pos not in self.untried_actions:
            self.untried_actions[pos] = [Action.MOVE_UP, Action.MOVE_RIGHT, Action.MOVE_DOWN, Action.MOVE_LEFT]
            
        self.backtracking = False
        
        # Try finding an unexplored path from current cell
        while self.untried_actions[pos]:
            nxt_act = self.untried_actions[pos].pop(0)
            
            # Predict
            nx, ny = pos
            if nxt_act == Action.MOVE_UP: ny -= 1
            elif nxt_act == Action.MOVE_DOWN: ny += 1
            elif nxt_act == Action.MOVE_LEFT: nx -= 1
            elif nxt_act == Action.MOVE_RIGHT: nx += 1
            
            if (nx, ny) not in self.visited and (pos, nxt_act) not in self.blocked_edges:
                self.last_action = nxt_act
                return [nxt_act]
                
        # If all paths tried, backtrack
        if len(self.path) > 1:
            self.path.pop() # Pop current
            prev = self.path[-1]
            bx, by = pos
            px, py = prev
            self.backtracking = True
            if py < by: self.last_action = Action.MOVE_UP
            elif py > by: self.last_action = Action.MOVE_DOWN
            elif px < bx: self.last_action = Action.MOVE_LEFT
            elif px > bx: self.last_action = Action.MOVE_RIGHT
            return [self.last_action]
            
        # Stuck at start or no path available
        return [Action.WAIT]
