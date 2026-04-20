# viz_path.py - Simple path visualization
from environment import MazeEnvironment
from ql_agent import QLearningAgent
from visualizer import Visualizer

print("Running Q-learning agent to find path...")

env = MazeEnvironment('maze_0.txt')
agent = QLearningAgent()
agent.epsilon = 0  # Use learned policy (no random moves)

env.reset()
agent.reset_episode()

last_result = None
done = False
turn_count = 0

while not done and turn_count < 5000:
    actions = agent.plan_turn(last_result)
    last_result = env.step(actions)
    turn_count += 1
    if last_result.is_goal_reached:
        done = True
        print(f"✅ Goal reached in {turn_count} turns!")
        print(f"   Path length: {len(agent.episode_path)} cells")

if done:
    # Visualize the path
    vis = Visualizer(env)
    vis.show_path(agent.episode_path)
    print("\n✅ Visualization saved to solution_path.png")
else:
    print("❌ Failed to reach goal")