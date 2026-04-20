from environment import MazeEnvironment
from ql_agent import QLearningAgent

print("=" * 50)
print("TESTING ON MAZE-BETA")
print("=" * 50)

# Load maze-beta (use maze_1.txt or maze_beta.txt)
env = MazeEnvironment('maze_1.txt')  # or 'maze_beta.txt'
agent = QLearningAgent()
agent.epsilon = 0  # No exploration, just follow learned policy

successes = 0
turns_list = []
deaths_total = 0

for episode in range(5):  # 5 episodes as required
    env.reset()
    agent.reset_episode()
    
    last_result = None
    turn_count = 0
    done = False
    
    while not done and turn_count < 10000:
        actions = agent.plan_turn(last_result)
        last_result = env.step(actions)
        turn_count += 1
        if last_result.is_goal_reached:
            done = True
    
    if last_result and last_result.is_goal_reached:
        successes += 1
        turns_list.append(turn_count)
        print(f"Episode {episode+1}: ✅ SUCCESS in {turn_count} turns")
    else:
        print(f"Episode {episode+1}: ❌ FAILED")
    
    deaths_total += env.deaths

print("\n" + "=" * 50)
print("MAZE-BETA RESULTS")
print("=" * 50)
print(f"Success Rate: {successes/5*100}%")
if turns_list:
    print(f"Average Turns: {sum(turns_list)/len(turns_list):.1f}")
    print(f"Average Path Length: {sum(turns_list)/len(turns_list):.1f}")
print(f"Death Rate: {deaths_total/(5*10000):.4f}")
print("=" * 50)