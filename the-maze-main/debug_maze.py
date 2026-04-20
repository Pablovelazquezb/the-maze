"""
Debug script to verify maze is solvable and find goal location
"""

from environment import MazeEnvironment
from dfs_agent import DFSAgent

def debug_maze():
    print("=" * 60)
    print("DEBUGGING MAZE-ALPHA")
    print("=" * 60)
    
    # Load the maze
    env = MazeEnvironment('maze_0.txt')
    
    print(f"\nStart position: {env.start_pos}")
    print(f"Goal position: {env.goal}")
    print(f"Number of pits: {len(env.pits)}")
    print(f"Number of teleports: {len(env.teleporters)}")
    print(f"Number of confusion pads: {len(env.confusion_pads)}")
    
    # Test if DFS can find the goal (to verify maze is solvable)
    print("\n" + "-" * 40)
    print("Testing DFS Agent (should find goal)...")
    
    agent = DFSAgent()
    env.reset()
    agent.reset_episode()
    
    done = False
    last_result = None
    turn_count = 0
    max_turns = 10000
    
    while not done and turn_count < max_turns:
        actions = agent.plan_turn(last_result)
        last_result = env.step(actions)
        
        if last_result.is_goal_reached:
            done = True
            print(f"\n✅ DFS SUCCESS! Goal found at turn {turn_count}")
            print(f"   Path length: {len(agent.path)} cells")
            break
        
        turn_count += 1
    
    if not done:
        print(f"\n❌ DFS also failed to find goal in {max_turns} turns")
        print("   This suggests the maze might be unsolvable or goal is unreachable")
    
    # Print a small map around start and goal
    print("\n" + "-" * 40)
    print("Map around START position:")
    sx, sy = env.start_pos
    for dy in range(-2, 3):
        line = ""
        for dx in range(-2, 3):
            x, y = sx + dx, sy + dy
            if 0 <= x < 64 and 0 <= y < 64:
                cell = env._get_cell(x, y)
                if (x, y) == env.start_pos:
                    line += "S"
                elif (x, y) == env.goal:
                    line += "G"
                else:
                    line += cell
            else:
                line += "X"
        print(f"  {line}")
    
    print("\nMap around GOAL position:")
    gx, gy = env.goal
    for dy in range(-2, 3):
        line = ""
        for dx in range(-2, 3):
            x, y = gx + dx, gy + dy
            if 0 <= x < 64 and 0 <= y < 64:
                cell = env._get_cell(x, y)
                if (x, y) == env.start_pos:
                    line += "S"
                elif (x, y) == env.goal:
                    line += "G"
                else:
                    line += cell
            else:
                line += "X"
        print(f"  {line}")


def test_q_agent_manually():
    """Manually test if Q agent can move at all"""
    print("\n" + "=" * 60)
    print("MANUAL Q-AGENT TEST")
    print("=" * 60)
    
    from ql_agent import QLearningAgent
    
    env = MazeEnvironment('maze_0.txt')
    agent = QLearningAgent()
    agent.epsilon = 0.0  # Greedy only
    
    env.reset()
    agent.reset_episode()
    
    print("\nTesting 10 steps of greedy actions:")
    last_result = None
    
    for step in range(10):
        actions = agent.plan_turn(last_result)
        last_result = env.step(actions)
        print(f"Step {step+1}: Pos={last_result.current_position}, "
              f"Action={actions[0].name if actions else 'None'}, "
              f"Wall hits={last_result.wall_hits}")
        
        if last_result.is_goal_reached:
            print("🎯 GOAL REACHED!")
            break


if __name__ == "__main__":
    debug_maze()
    test_q_agent_manually()