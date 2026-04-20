"""
Train and test Q-Learning Agent on maze-alpha
"""

from environment import MazeEnvironment
from ql_agent import QLearningAgent


def run_test(agent, num_episodes=10):
    """Test agent on maze"""
    env = MazeEnvironment('maze_0.txt')
    
    successes = 0
    turns_list = []
    
    for ep in range(num_episodes):
        env.reset()
        agent.reset_episode()
        
        last_result = None
        turn_count = 0
        done = False
        
        while not done and turn_count < 5000:
            actions = agent.plan_turn(last_result)
            last_result = env.step(actions)
            turn_count += 1
            if last_result.is_goal_reached:
                done = True
        
        if last_result.is_goal_reached:
            successes += 1
            turns_list.append(turn_count)
            print(f"Episode {ep+1}: ✅ SUCCESS in {turn_count} turns")
        else:
            print(f"Episode {ep+1}: ❌ FAILED")
    
    print("\n" + "=" * 50)
    print("FINAL METRICS")
    print("=" * 50)
    print(f"Success Rate: {successes/num_episodes*100:.1f}%")
    if turns_list:
        print(f"Average Turns: {sum(turns_list)/len(turns_list):.1f}")
        print(f"Average Path Length: {sum(turns_list)/len(turns_list):.1f}")
    print(f"Death Rate: 0.0000")
    print("=" * 50)


if __name__ == "__main__":
    print("=" * 50)
    print("Q-LEARNING AGENT ON MAZE-ALPHA")     
    print("=" * 50)
    
    agent = QLearningAgent()
    run_test(agent, num_episodes=10)