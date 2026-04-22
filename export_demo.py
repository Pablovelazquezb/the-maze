"""
Export Demo Data — COSC 4368 Spring 2026
Runs the trained Q-Learning agent on maze_1.txt (Maze Beta)
and exports the maze structure, hazards, and bot's path to JSON
for the web presentation.
"""

import json
from environment import MazeEnvironment
from ql_agent import QLearningAgent

def export_maze_data(maze_file, output_file, max_turns=5000):
    print(f"Loading environment {maze_file}...")
    env = MazeEnvironment(maze_file, max_turns=max_turns)
    
    agent = QLearningAgent()
    if not agent.load('q_table_alpha.pkl'):
        print("ERROR: Could not load trained agent! Please run run_checkin3.py first.")
        return

    # Disable learning for the demo
    agent.alpha = 0.0
    agent.epsilon = 0.05
    
    print("Running simulation...")
    env.reset()
    agent.reset_episode()

    last_result = None
    turn = 0
    done = False
    
    # Track the exact positions
    path = [env.start_pos]
    actions_taken = []
    
    while not done and turn < max_turns:
        actions = agent.plan_turn(last_result)
        last_result = env.step(actions)
        
        # Log the step
        path.append(last_result.current_position)
        # Note: if actions was a list, just log what was planned
        if actions:
            actions_taken.append(actions[0].name)
            
        turn += 1
        
        if last_result.is_goal_reached:
            agent.plan_turn(last_result)
            done = True
        elif last_result.is_dead:
            agent.plan_turn(last_result)

    if not done:
        print("WARNING: Agent did not reach goal in demo run.")
    
    # Extract walls
    walls = []
    for y in range(64):
        for x in range(64):
            if env._get_cell(x, y) == 'X':
                walls.append([x, y])

    data = {
        'maze_name': maze_file,
        'dimensions': [64, 64],
        'start_pos': env.start_pos,
        'goal_pos': env.goal,
        'walls': walls,
        'pits': env.pits,
        'teleporters': env.teleporters,
        'confusion_pads': env.confusion_pads,
        'path': path,
        'turns': turn,
        'success': done
    }
    
    with open(output_file, 'w') as f:
        json.dump(data, f)
        
    print(f"Successfully exported demo data to {output_file}")
    print(f"Path length: {len(path)} steps")

if __name__ == "__main__":
    export_maze_data('maze_1.txt', 'demo_data.json')
