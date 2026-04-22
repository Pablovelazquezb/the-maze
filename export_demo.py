"""
Export Demo Data — COSC 4368 Spring 2026
Runs the trained Q-Learning agent on maze_1.txt (Maze Beta) and maze_2.txt (Maze Gamma)
and exports the maze structure, hazards, and bot's path to JSON
for the web presentation.
"""

import json
from environment import MazeEnvironment
from ql_agent import QLearningAgent
from metrics import MetricsCalculator

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
    calc = MetricsCalculator()
    
    print(f"Running simulation on {maze_file}...")
    
    # Run 20 episodes to get metrics
    for ep in range(1, 21):
        env.reset()
        agent.reset_episode()

        last_result = None
        turn = 0
        done = False
        total_visits = 0
        
        # Track the exact positions for the FIRST episode only (for the visualizer)
        path = [env.start_pos]
        
        while not done and turn < max_turns:
            actions = agent.plan_turn(last_result)
            last_result = env.step(actions)
            
            if ep == 1:
                path.append(last_result.current_position)
                
            turn += 1
            total_visits += 1
            
            if last_result.is_goal_reached:
                agent.plan_turn(last_result)
                done = True
            elif last_result.is_dead:
                agent.plan_turn(last_result)

        stats = env.get_episode_stats()
        calc.add_episode(
            success=stats['goal_reached'],
            turns=stats['turns_taken'],
            path_length=len(agent.episode_path) if ep==1 else stats['turns_taken'],
            deaths=stats['deaths'],
            unique_cells=stats['cells_explored'],
            total_visits=total_visits,
            replanning_count=getattr(agent, 'replan_count', 0)
        )
        
        if ep == 1:
            demo_path = path
            demo_turns = turn
            demo_done = done

    if not demo_done:
        print(f"WARNING: Agent did not reach goal in demo run for {maze_file}.")
    
    # Extract walls
    walls = []
    for y in range(64):
        for x in range(64):
            if env._get_cell(x, y) == 'X':
                walls.append([x, y])

    metrics = calc.get_all()

    data = {
        'maze_name': maze_file,
        'dimensions': [64, 64],
        'start_pos': env.start_pos,
        'goal_pos': env.goal,
        'walls': walls,
        'pits': env.pits,
        'teleporters': env.teleporters,
        'confusion_pads': env.confusion_pads,
        'path': demo_path,
        'turns': demo_turns,
        'success': demo_done,
        'metrics': metrics
    }
    
    with open(output_file, 'w') as f:
        json.dump(data, f)
        
    print(f"Successfully exported {maze_file} to {output_file}")
    print(f"Metrics: Success={metrics['success_rate']*100}% AvgTurns={metrics['avg_turns_to_solution']}")

if __name__ == "__main__":
    export_maze_data('maze_1.txt', 'presentation/src/assets/demo_data.json')
    export_maze_data('maze_2.txt', 'presentation/src/assets/demo_data_gamma.json')
