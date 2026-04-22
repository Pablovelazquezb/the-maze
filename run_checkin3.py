"""
Check-In #3 FINAL Runner — COSC 4368 Spring 2026
Trains Q-Learning on maze-alpha, tests on maze-beta, reports all metrics.
Generates training curves and solution visualizations.
"""

import json, time, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from environment import MazeEnvironment, Action
from ql_agent import QLearningAgent
from metrics import MetricsCalculator


def run_episode(env, agent, max_turns=10000):
    """Run one episode and return stats."""
    env.reset()
    agent.reset_episode()

    last_result = None
    turn = 0
    done = False
    total_visits = 0

    while not done and turn < max_turns:
        actions = agent.plan_turn(last_result)
        last_result = env.step(actions)
        turn += 1
        total_visits += 1

        if last_result.is_goal_reached:
            # Let agent learn the goal reward
            agent.plan_turn(last_result)
            done = True
        elif last_result.is_dead:
            # Let agent learn from death
            agent.plan_turn(last_result)

    stats = env.get_episode_stats()
    return {
        'success': stats['goal_reached'],
        'turns': stats['turns_taken'],
        'path_length': len(agent.episode_path),
        'deaths': stats['deaths'],
        'unique_cells': stats['cells_explored'],
        'total_visits': total_visits,
        'replanning_count': getattr(agent, 'replan_count', 0),
    }


def train_on_alpha(maze_file='maze_0.txt', num_episodes=50, max_turns=10000):
    """Train the Q-Learning agent on maze-alpha."""
    print("\n" + "="*60)
    print("  PHASE 1: TRAINING ON MAZE-ALPHA")
    print("="*60)

    env = MazeEnvironment(maze_file, max_turns=max_turns)
    agent = QLearningAgent()
    calc = MetricsCalculator()

    print(f"  Maze: {maze_file}")
    print(f"  Start: {env.start_pos}  Goal: {env.goal}")
    print(f"  Pits: {len(env.pits)}  Teleporters: {len(env.teleporters)}  Confusion: {len(env.confusion_pads)}")
    print(f"  Episodes: {num_episodes}")
    print()

    training_log = {'episode': [], 'success': [], 'turns': [], 'deaths': [],
                    'epsilon': [], 'cumulative_success_rate': []}
    successes_so_far = 0

    t0 = time.time()
    for ep in range(1, num_episodes + 1):
        stats = run_episode(env, agent, max_turns)
        calc.add_episode(**stats)

        if stats['success']:
            successes_so_far += 1

        training_log['episode'].append(ep)
        training_log['success'].append(stats['success'])
        training_log['turns'].append(stats['turns'])
        training_log['deaths'].append(stats['deaths'])
        training_log['epsilon'].append(agent.epsilon)
        training_log['cumulative_success_rate'].append(successes_so_far / ep)

        status = "✅" if stats['success'] else "❌"
        print(f"  Ep {ep:3d}/{num_episodes} {status}  turns={stats['turns']:5d}  "
              f"deaths={stats['deaths']}  path={stats['path_length']:5d}  ε={agent.epsilon:.3f}")

    elapsed = time.time() - t0
    print(f"\n  Training complete in {elapsed:.1f}s")

    # Save agent
    agent.save('q_table_alpha.pkl')
    print("  Saved trained agent → q_table_alpha.pkl")

    # Print metrics
    calc.print_summary("MAZE-ALPHA (Training)")
    calc.save_json('metrics_alpha.json')

    # Generate training curve plot
    _plot_training_curve(training_log, 'training_curve_alpha.png')

    return agent, calc, training_log


def test_on_beta(agent, maze_file='maze_1.txt', num_episodes=20, max_turns=10000):
    """Test the trained agent on maze-beta WITHOUT further training."""
    print("\n" + "="*60)
    print("  PHASE 2: TESTING ON MAZE-BETA (NO TRAINING)")
    print("="*60)

    env = MazeEnvironment(maze_file, max_turns=max_turns)
    calc = MetricsCalculator()

    print(f"  Maze: {maze_file}")
    print(f"  Start: {env.start_pos}  Goal: {env.goal}")
    print(f"  Pits: {len(env.pits)}  Teleporters: {len(env.teleporters)}  Confusion: {len(env.confusion_pads)}")
    print(f"  Episodes: {num_episodes}")

    # Freeze learning: set epsilon to 0 and alpha to 0 for pure exploitation
    # But keep exploration ability since walls might differ
    old_epsilon = agent.epsilon
    old_alpha = agent.alpha
    agent.epsilon = 0.1  # Small exploration for new maze
    agent.alpha = 0.0    # NO learning on beta

    for ep in range(1, num_episodes + 1):
        stats = run_episode(env, agent, max_turns)
        calc.add_episode(**stats)

        status = "✅" if stats['success'] else "❌"
        print(f"  Ep {ep:3d}/{num_episodes} {status}  turns={stats['turns']:5d}  "
              f"deaths={stats['deaths']}  path={stats['path_length']:5d}")

    # Restore
    agent.epsilon = old_epsilon
    agent.alpha = old_alpha

    calc.print_summary("MAZE-BETA (Zero-Shot Test)")
    calc.save_json('metrics_beta.json')

    return calc


def _plot_training_curve(log, filename):
    """Generate training curve visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Q-Learning Training on Maze-Alpha', fontsize=16, fontweight='bold')

    eps = log['episode']

    # 1. Cumulative success rate
    ax = axes[0, 0]
    ax.plot(eps, [s*100 for s in log['cumulative_success_rate']], 'g-', linewidth=2)
    ax.set_title('Cumulative Success Rate')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate (%)')
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)

    # 2. Turns per episode
    ax = axes[0, 1]
    ax.plot(eps, log['turns'], 'b-', alpha=0.5, linewidth=1)
    # Rolling average
    window = max(3, len(eps) // 10)
    if len(eps) >= window:
        rolling = np.convolve(log['turns'], np.ones(window)/window, mode='valid')
        ax.plot(eps[window-1:], rolling, 'r-', linewidth=2, label=f'{window}-ep avg')
        ax.legend()
    ax.set_title('Turns per Episode')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Turns')
    ax.grid(True, alpha=0.3)

    # 3. Deaths per episode
    ax = axes[1, 0]
    ax.bar(eps, log['deaths'], color='red', alpha=0.6)
    ax.set_title('Deaths per Episode')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Deaths')
    ax.grid(True, alpha=0.3)

    # 4. Epsilon decay
    ax = axes[1, 1]
    ax.plot(eps, log['epsilon'], 'purple', linewidth=2)
    ax.set_title('Epsilon (Exploration Rate)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Epsilon')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved training curve → {filename}")


def visualize_solution(agent, maze_file, output_file='solution_path.png'):
    """Run one episode and visualize the solution path."""
    env = MazeEnvironment(maze_file)
    agent.epsilon = 0.0  # Pure exploitation

    env.reset()
    agent.reset_episode()

    last_result = None
    path_positions = [env.start_pos]
    turn = 0

    while turn < 10000:
        actions = agent.plan_turn(last_result)
        last_result = env.step(actions)
        path_positions.append(last_result.current_position)
        turn += 1
        if last_result.is_goal_reached:
            agent.plan_turn(last_result)
            break

    if not last_result or not last_result.is_goal_reached:
        print("  ⚠️  Could not generate solution visualization (agent failed)")
        return

    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(14, 14))

    # Draw maze grid
    grid = np.ones((64, 64, 3))  # white background

    # Mark walls based on agent's knowledge
    for y in range(64):
        for x in range(64):
            cell = env._get_cell(x, y)
            if cell == 'X':
                grid[y, x] = [0, 0, 0]

    ax.imshow(grid, interpolation='nearest')

    # Draw path
    xs = [p[0] for p in path_positions]
    ys = [p[1] for p in path_positions]
    ax.plot(xs, ys, 'b-', linewidth=1.5, alpha=0.6, label=f'Path ({len(path_positions)} steps)')

    # Mark start and goal
    sx, sy = env.start_pos
    gx, gy = env.goal
    ax.plot(sx, sy, 'go', markersize=12, label='Start', zorder=5)
    ax.plot(gx, gy, 'r*', markersize=15, label='Goal', zorder=5)

    # Mark hazards
    for p in env.pits:
        ax.plot(p[0], p[1], 'rv', markersize=6, alpha=0.7)
    for t in env.teleporters:
        ax.plot(t[0], t[1], 'bs', markersize=6, alpha=0.7)
    for c in env.confusion_pads:
        ax.plot(c[0], c[1], 'y^', markersize=6, alpha=0.7)

    ax.set_title(f'Solution Path — {os.path.basename(maze_file)} ({turn} turns)', fontsize=14)
    ax.legend(loc='upper right')
    ax.set_xlim(-0.5, 63.5)
    ax.set_ylim(63.5, -0.5)
    ax.grid(True, alpha=0.1)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved solution → {output_file}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("="*60)
    print("  SILENT CARTOGRAPHER — CHECK-IN #3 FINAL")
    print("  Q-Learning Maze Solver with Hazard Avoidance")
    print("  COSC 4368 — Spring 2026 — Group 14")
    print("="*60)

    # Phase 1: Train on maze-alpha (maze_0.txt with hazards injected)
    # The environment injects hazards for any maze with 'alpha' or 'maze_1' in name
    agent, alpha_metrics, train_log = train_on_alpha(
        maze_file='maze_0.txt',
        num_episodes=40,
        max_turns=25000
    )

    # Visualize alpha solution
    visualize_solution(agent, 'maze_0.txt', 'solution_alpha.png')

    # Phase 2: Test on maze-beta (NO training)
    beta_metrics = test_on_beta(
        agent,
        maze_file='maze_1.txt',
        num_episodes=20,
        max_turns=12000
    )

    # Visualize beta solution
    visualize_solution(agent, 'maze_1.txt', 'solution_beta.png')

    # Final summary
    print("\n" + "="*60)
    print("  FINAL SUMMARY")
    print("="*60)

    am = alpha_metrics.get_all()
    bm = beta_metrics.get_all()

    print(f"\n  {'Metric':<30} {'Alpha':>10} {'Beta':>10}")
    print(f"  {'─'*52}")
    print(f"  {'Success Rate':<30} {am['success_rate']*100:9.1f}% {bm['success_rate']*100:9.1f}%")
    print(f"  {'Avg Path Length':<30} {am['avg_path_length']:10.1f} {bm['avg_path_length']:10.1f}")
    print(f"  {'Avg Turns to Solution':<30} {am['avg_turns_to_solution']:10.1f} {bm['avg_turns_to_solution']:10.1f}")
    print(f"  {'Death Rate':<30} {am['death_rate']:10.4f} {bm['death_rate']:10.4f}")
    print(f"  {'Exploration Efficiency':<30} {am['exploration_efficiency']:10.3f} {bm['exploration_efficiency']:10.3f}")
    print(f"  {'Map Completeness':<30} {am['map_completeness']*100:9.1f}% {bm['map_completeness']*100:9.1f}%")

    print("\n  Generated files:")
    for f in ['q_table_alpha.pkl', 'metrics_alpha.json', 'metrics_beta.json',
              'training_curve_alpha.png', 'solution_alpha.png', 'solution_beta.png']:
        if os.path.exists(f):
            print(f"    ✅ {f} ({os.path.getsize(f):,} bytes)")
        else:
            print(f"    ❌ {f} (missing)")

    print("\n" + "="*60)
    print("  CHECK-IN #3 COMPLETE")
    print("="*60)
