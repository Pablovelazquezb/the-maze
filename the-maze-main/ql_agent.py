"""
Q-Learning Agent that learns from DFS path
COSC 4368 - Spring 2026
"""

import random
import pickle
from collections import defaultdict
from environment import Action, Agent


class QLearningAgent(Agent):
    def __init__(self, alpha=0.3, gamma=0.9, epsilon=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Q-table: (x, y, action) -> value
        self.q_table = defaultdict(float)
        
        # Tracking
        self.last_state = None
        self.last_action = None
        self.current_pos = None
        self.visited = set()
        self.episode_path = []  # ADDED THIS - stores path for visualization
        
        self.actions = [Action.MOVE_UP, Action.MOVE_DOWN, 
                        Action.MOVE_LEFT, Action.MOVE_RIGHT]
        
        # Train on DFS path if available
        self.train_on_dfs_path()
    
    def train_on_dfs_path(self):
        """Train Q-values using the successful DFS path"""
        try:
            from dfs_agent import DFSAgent
            from environment import MazeEnvironment
            
            print("Training Q-learning on DFS path...")
            env = MazeEnvironment('maze_0.txt')
            dfs = DFSAgent()
            
            env.reset()
            dfs.reset_episode()
            
            # Run DFS to get path
            done = False
            last_result = None
            path = []
            
            while not done and len(path) < 10000:
                actions = dfs.plan_turn(last_result)
                last_result = env.step(actions)
                path.append(last_result.current_position)
                if last_result.is_goal_reached:
                    done = True
            
            if done:
                print(f"DFS found goal in {len(path)} steps. Training Q-table...")
                
                # Train Q-values along the path
                for i in range(len(path) - 1):
                    current = path[i]
                    next_pos = path[i + 1]
                    
                    # Determine action taken
                    dx = next_pos[0] - current[0]
                    dy = next_pos[1] - current[1]
                    
                    if dx == 1:
                        action = 3  # RIGHT
                    elif dx == -1:
                        action = 2  # LEFT
                    elif dy == 1:
                        action = 1  # DOWN
                    elif dy == -1:
                        action = 0  # UP
                    else:
                        continue
                    
                    # High reward for following successful path
                    self.q_table[(current[0], current[1], action)] = 100
                
                print(f"Trained on {len(path)} steps")
        except Exception as e:
            print(f"DFS training skipped: {e}")
    
    def get_q(self, x, y, action):
        return self.q_table[(x, y, action)]
    
    def choose_action(self, x, y):
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        
        best_action = 0
        best_value = self.get_q(x, y, 0)
        for a in range(1, 4):
            val = self.get_q(x, y, a)
            if val > best_value:
                best_value = val
                best_action = a
        return best_action
    
    def plan_turn(self, last_result):
        if last_result is None:
            self.last_state = None
            self.visited.clear()
            self.episode_path = []  # Reset path
            return [Action.MOVE_RIGHT]
        
        self.current_pos = last_result.current_position
        self.visited.add(self.current_pos)
        self.episode_path.append(self.current_pos)  # Record path
        
        if last_result.is_goal_reached:
            return [Action.WAIT]
        
        x, y = self.current_pos
        action_idx = self.choose_action(x, y)
        
        # Update Q-learning if we have previous state
        if self.last_state is not None:
            lx, ly = self.last_state
            reward = -0.1
            if self.current_pos not in self.visited:
                reward = 1.0
            if last_result.wall_hits > 0:
                reward = -2.0
            
            old_q = self.get_q(lx, ly, self.last_action)
            new_q = old_q + self.alpha * (reward + self.gamma * self.get_q(x, y, action_idx) - old_q)
            self.q_table[(lx, ly, self.last_action)] = new_q
        
        self.last_state = (x, y)
        self.last_action = action_idx
        
        return [self.actions[action_idx]]
    
    def reset_episode(self):
        self.last_state = None
        self.last_action = None
        self.episode_path = []  # Reset path
    
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
    
    def load(self, filename):
        try:
            with open(filename, 'rb') as f:
                self.q_table = defaultdict(float, pickle.load(f))
            return True
        except:
            return False
        
# """
# Q-Learning Agent that learns from DFS path
# COSC 4368 - Spring 2026
# """

# import random
# import pickle
# from collections import defaultdict
# from environment import Action, Agent


# class QLearningAgent(Agent):
#     def __init__(self, alpha=0.3, gamma=0.9, epsilon=0.1):
#         super().__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.epsilon = epsilon
        
#         # Q-table: (x, y, action) -> value
#         self.q_table = defaultdict(float)
        
#         # Tracking
#         self.last_state = None
#         self.last_action = None
#         self.current_pos = None
#         self.visited = set()
        
#         self.actions = [Action.MOVE_UP, Action.MOVE_DOWN, 
#                         Action.MOVE_LEFT, Action.MOVE_RIGHT]
        
#         # Train on DFS path if available
#         self.train_on_dfs_path()
    
#     def train_on_dfs_path(self):
#         """Train Q-values using the successful DFS path"""
#         try:
#             from dfs_agent import DFSAgent
#             from environment import MazeEnvironment
            
#             print("Training Q-learning on DFS path...")
#             env = MazeEnvironment('maze_0.txt')
#             dfs = DFSAgent()
            
#             env.reset()
#             dfs.reset_episode()
            
#             # Run DFS to get path
#             done = False
#             last_result = None
#             path = []
            
#             while not done and len(path) < 10000:
#                 actions = dfs.plan_turn(last_result)
#                 last_result = env.step(actions)
#                 path.append(last_result.current_position)
#                 if last_result.is_goal_reached:
#                     done = True
            
#             if done:
#                 print(f"DFS found goal in {len(path)} steps. Training Q-table...")
                
#                 # Train Q-values along the path
#                 for i in range(len(path) - 1):
#                     current = path[i]
#                     next_pos = path[i + 1]
                    
#                     # Determine action taken
#                     dx = next_pos[0] - current[0]
#                     dy = next_pos[1] - current[1]
                    
#                     if dx == 1:
#                         action = 3  # RIGHT
#                     elif dx == -1:
#                         action = 2  # LEFT
#                     elif dy == 1:
#                         action = 1  # DOWN
#                     elif dy == -1:
#                         action = 0  # UP
#                     else:
#                         continue
                    
#                     # High reward for following successful path
#                     self.q_table[(current[0], current[1], action)] = 100
                
#                 print(f"Trained on {len(path)} steps")
#         except Exception as e:
#             print(f"DFS training skipped: {e}")
    
#     def get_q(self, x, y, action):
#         return self.q_table[(x, y, action)]
    
#     def choose_action(self, x, y):
#         if random.random() < self.epsilon:
#             return random.randint(0, 3)
        
#         best_action = 0
#         best_value = self.get_q(x, y, 0)
#         for a in range(1, 4):
#             val = self.get_q(x, y, a)
#             if val > best_value:
#                 best_value = val
#                 best_action = a
#         return best_action
    
#     def plan_turn(self, last_result):
#         if last_result is None:
#             self.last_state = None
#             self.visited.clear()
#             return [Action.MOVE_RIGHT]
        
#         self.current_pos = last_result.current_position
#         self.visited.add(self.current_pos)
        
#         if last_result.is_goal_reached:
#             return [Action.WAIT]
        
#         x, y = self.current_pos
#         action_idx = self.choose_action(x, y)
        
#         # Update Q-learning if we have previous state
#         if self.last_state is not None:
#             lx, ly = self.last_state
#             reward = -0.1
#             if self.current_pos not in self.visited:
#                 reward = 1.0
#             if last_result.wall_hits > 0:
#                 reward = -2.0
            
#             old_q = self.get_q(lx, ly, self.last_action)
#             new_q = old_q + self.alpha * (reward + self.gamma * self.get_q(x, y, action_idx) - old_q)
#             self.q_table[(lx, ly, self.last_action)] = new_q
        
#         self.last_state = (x, y)
#         self.last_action = action_idx
        
#         return [self.actions[action_idx]]
    
#     def reset_episode(self):
#         self.last_state = None
#         self.last_action = None
    
#     def save(self, filename):
#         with open(filename, 'wb') as f:
#             pickle.dump(dict(self.q_table), f)
    
#     def load(self, filename):
#         try:
#             with open(filename, 'rb') as f:
#                 self.q_table = defaultdict(float, pickle.load(f))
#             return True
#         except:
#             return False