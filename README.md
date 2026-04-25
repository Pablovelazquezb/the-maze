# Silent Cartographer — Maze Navigation Agent
**COSC 4368 AI Spring 2026 | Group 14**

An intelligent agent that navigates unknown 64×64 mazes containing death pits, teleport pads, and confusion traps using systematic BFS frontier exploration.

---

## Project Structure

```
.
├── agent.py            # NaiveAgent — BFS frontier explorer
├── environment.py      # Local maze simulation (walls, pits, teleports, confusion)
├── maze_parser.py      # PNG-to-grid parser for maze images
├── visualizer.py       # Agent map and path visualization
├── report.md           # Project progress report
├── results_naive_agent.md  # 5-episode evaluation results
├── maze-alpha/         # Training maze (MAZE_0.png, MAZE_1.png, MAZE_2.png)
├── maze-gamma/         # Testing maze
└── output/             # Generated visualizations
```

---

## Requirements

- Python 3.10+
- `numpy`
- `Pillow`
- `matplotlib`

Install dependencies:

```bash
pip install numpy Pillow matplotlib
```

---

## Running the Agent

### Single episode (training maze)

```bash
python agent.py
```

Output:
```
Running NaiveAgent on training maze...
  turn=    0  pos=(32,62)  explored=   2/4096  deaths=0  confused=0
  turn=  500  pos=(27,39)  explored= 219/4096  ...
  *** GOAL reached on turn 4286!

Episode stats: {'turns_taken': 4286, 'deaths': 4, 'confused': 2, 'cells_explored': 1785, 'goal_reached': True}
```

### Multiple episodes (benchmark)

Run 5 episodes with persistent memory:

```python
from environment import MazeEnvironment
from agent import NaiveAgent

env   = MazeEnvironment("training")   # or "testing"
start = env.reset()
agent = NaiveAgent(start)

for episode in range(1, 6):
    env.reset()
    agent.reset_episode()      # resets position, keeps map memory
    last_result = None

    for turn in range(10_000):
        actions = agent.plan_turn(last_result)
        last_result = env.step(actions)
        if last_result.is_goal_reached:
            break

    stats = env.get_episode_stats()
    print(f"Episode {episode}: {stats}")
```

### Visualizer

```python
from visualizer import Visualizer
from agent import NaiveAgent
from environment import MazeEnvironment

viz   = Visualizer()
env   = MazeEnvironment("training")
start = env.reset()
agent = NaiveAgent(start)

# Run one episode
last_result = None
for _ in range(10_000):
    actions = agent.plan_turn(last_result)
    last_result = env.step(actions)
    if last_result.is_goal_reached:
        break

# Save agent's internal map
viz.visualize_map(agent.memory, save_path="output/agent_map.png")
```

---

## How the Agent Works

The `NaiveAgent` uses **systematic BFS frontier exploration** in three modes:

| Mode | Trigger | Behavior |
|------|---------|----------|
| **PROBE** | Unexplored edge at current cell | Send exactly 1 action; record edge type and cell type from result |
| **NAVIGATE** | No unexplored edges at current cell | BFS to nearest frontier; batch up to 5 steps through confirmed-safe cells |
| **GOAL** | Goal cell known | BFS directly to goal |

### World Model (`agent.memory`)

| Key | Type | Description |
|-----|------|-------------|
| `open_edges` | `set[frozenset]` | Confirmed passable cell boundaries |
| `closed_edges` | `set[frozenset]` | Confirmed walls |
| `probed` | `set[(col,row)]` | Cells confirmed non-confusion (safe to batch through) |
| `hazards` | `dict` | Maps `(col,row)` → cell type (pit, confusion, goal, teleport) |
| `teleport_map` | `dict` | Maps teleport source → destination |
| `goal` | `(col,row)` or `None` | Goal location once discovered |

### Hazard Handling

- **Death pits** — Recorded on death; permanently excluded from BFS navigation paths
- **Teleport pads** — Discovered on first step; source cell excluded from BFS so the agent is never unintentionally teleported
- **Confusion traps** — Identified via single-step probe; actions are inverted for the affected turns using a mirrored confusion counter
- **Teleport isolation** — When stranded in a disconnected component, the agent intentionally steps into a known pit to respawn at start

---

## Performance Results (maze-alpha, 5 episodes)

| Episode | Turns | Deaths | Goal |
|---------|------:|-------:|:----:|
| 1       | 4,286 |      4 | ✅   |
| 2       | 1,788 |      1 | ✅   |
| 3       |   194 |      0 | ✅   |
| 4       |    87 |      0 | ✅   |
| 5       |    87 |      0 | ✅   |

| Metric        | NaiveAgent | Expected | Stretch |
|---------------|:----------:|:--------:|:-------:|
| Success rate  | **100%**   | >80%     | >95%    |
| Avg turns     | **1,288**  | <1,000   | <500    |
| Death rate    | **0.00078**| <0.050   | <0.010  |

Memory persists across episodes — by episode 4 the agent navigates directly to the goal in **87 turns**.

---

## Output Files

| File | Description |
|------|-------------|
| `output/maze_alpha_ground_truth.png` | Full ground-truth map of maze-alpha with all hazards |
| `output/maze_gamma_ground_truth.png` | Full ground-truth map of maze-gamma |
| `output/agent_map_demo.png` | Agent's internal map after episode 1 exploration |
| `output/episode_path_demo.png` | Path taken during episode 1 |

---

## AI Tool Disclosure

This project used **Claude (Anthropic)** for debugging assistance, algorithm design discussion, and code review. All algorithmic design decisions, including the probe/navigate state machine, confusion counter logic, and teleport recording fix, were developed and validated by the team.
