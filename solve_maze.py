from environment import MazeEnvironment
from dfs_agent import DFSAgent
from visualizer import Visualizer

def solve():
    print("Loading Maze without hazards (MAZE 0)...")
    env = MazeEnvironment('maze_0.txt')
    agent = DFSAgent()
    
    pos = env.reset()
    agent.reset_episode()
    done = False
    turn_count = 0
    last_result = None
    
    print("Navigating...")
    while not done and turn_count < 10000:
        actions = agent.plan_turn(last_result)
        last_result = env.step(actions)
        
        if last_result.is_goal_reached:
            print(f"Goal reached in {turn_count} turns!")
            done = True
            
        turn_count += 1
        
    stats = env.get_episode_stats()
    print("Final Stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    print("Generating visualization...")
    vis = Visualizer(env)
    vis.show_path(agent.path)

if __name__ == "__main__":
    solve()
