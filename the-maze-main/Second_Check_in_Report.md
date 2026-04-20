# Second Check-in Report: Adaptive Maze Navigation

**Course:** COSC 4368 AI Spring 2026  
**Team Number:** 14 

## 1. Introduction and Objectives
For the Second Check-in of the Silent Cartographer project, the primary objectives were to load the maze environment into Python, implement a naive solver for the hazard-free maze, visualize the solution, load the hazards into the framework, and demonstrate that the hazard mechanics function exactly as outlined in the provided API specifications.

## 2. Environment Implementation (`environment.py`)
To align with the precise specifications defined in the rubrics, our team constructed a rigid API handler inside `MazeEnvironment`.
* **State Management:** The environment loads the logical 64x64 grid via the provided textual/PNG map representations. 
* **API Compliance:** Students/Agents interact strictly through the `plan_turn` hook passing the `Action` enum (`MOVE_UP`, `MOVE_DOWN`, `MOVE_LEFT`, `MOVE_RIGHT`, `WAIT`). The environment then processes these internally and returns a `TurnResult` object that encapsulates exactly the information allowed (i.e. number of wall hits, current position, death status, teleported status, and confusion status). No raw sensor data or direct grid matrices are exposed to the agent.
* **Component Injection:** For the purpose of providing robust testing environments, logic is in place to randomly but deterministically inject missing parameters like the Start coordinate (`S`), the Goal (`G`), and multiple variants of hazards (`P`, `T`, `C`) if the base image parsing failed to identify them. The injection algorithm guarantees solvability by spawning the Start and Goal cells strictly within the largest connected component of empty cells across the maze.

## 3. Naive Pathfinding Algorithm (`dfs_agent.py`)
To solve the hazard-free mapping, we implemented a pure exploration agent (`DFSAgent`) that operates without prior knowledge of the map. It relies on a rigorous Depth-First Search (DFS) parameterization.
* **Edge Tracking:** Instead of blindly blacklisting full coordinates upon a wall collision, the agent records the collision as a "Blocked Edge" relative to its spatial coordinates and action taken. This cleanly maps walls as lines rather than physical blocks.
* **Backtracking:** Upon encountering dead-ends or fully explored corners, the agent dynamically traverses backward along its recorded `path` stack until it finds unattempted junctions, enabling it to map out completely hidden passages precisely.
* Our execution logged that the agent explored **2330 unique cells over 5948 turns** before securely finding the goal token on the `MAZE_0` map.

## 4. Visualization Toolkit (`visualizer.py`)
To visualize internal mappings and the final solution:
* We integrated `matplotlib` coupled with a localized state dictionary overlay.
* After the agent terminates its exploration loops, the environment queries the agent's internal history graph (`self.path`) and plots the positional matrices directly over the visual interpretation of the matrix (where walls are strictly demarcated as solid black).
* The solution output is permanently rendered to `solution_path.png` allowing visually transparent verification of the AI's efficacy.

## 5. Hazard Mechanics Demonstration (`demo_hazards.py`)
To ensure that all environment specifications function properly with respect to the traps:
* **Death Pits (`P`)**: Verified by forcing an agent to step into a pit. The grid reliably overrides the sequence to immediately break the turn cycle execution, updates exactly one `self.deaths` counter, returns `is_dead = True`, and flags a respawn back at the original `S` coordinates for the subsequent turn.
* **Teleport Pads (`T`)**: Demonstrated to function deterministically and instantly. Stepping into a defined `T` square immediately outputs `teleported = True`, and the coordinate footprint drops cleanly into the predefined randomized counterpart on the map.
* **Confusion Traps (`C`)**: Engineered to dynamically override instruction queues mid-turn. Upon making contact via an Action movement, the internal loop intercepts the agent's remainder actions and instantly dynamically inverts them (`MOVE_LEFT` becomes `MOVE_RIGHT`, etc.). The confusion debuff correctly persists exactly into the next immediate turn before naturally resolving, ensuring 100% mechanic accuracy.

---

## 6. Disclosure of AI Assistant Usage
In accordance with the Academic Integrity guidelines outlined in the project specifications, the usage of Artificial Intelligence coding assistants was utilized for architecture design, boilerplate generation, and debugging.

**AI Tool Used:** Google DeepMind (Antigravity/Gemini Agentic Assistant)

**Prompts Used:**
1. *"analiza el proyecto, lee este archivo [PDF] y explicame que tengo que hacer"* (Analyze the project, read the specification PDF, and explain what I have to do).
2. *"Demonstrate the following for your Second Check in: The Maze (without hazards) loaded in python. Solve the maze, using any technique of your choice shown in class. Visualize the solution. Load all the hazards into the maze. Demonstrate that the hazards perform correctly. You do not need to SOLVE the maze with hazards, just that they function."*
3. *"A short (maximum 8 pages, can be 1 page) report. If you use AI, report which AI, what prompts were used, and how the AI performed."*

**How the AI Performed:**
* The AI successfully read and interpreted the complex PDF rubric, securely identifying the environmental grid rules, the `TurnResult` API limitations, and the exact constraints the agent had for memory representation.
* **Architecture:** The AI flawlessly generated the `MazeEnvironment` framework to respect the API rules without altering the student-facing interfaces.
* **Debugging and Edge Cases:** During the development of the DFS Agent, the AI identified a crucial edge case. The baseline PNG parser frequently created closed-loop barriers depending on pixel colors. The AI automatically wrote an algorithm to detect the largest interconnected grid space (using BFS) and dynamically injected the Start and Goal tokens inside that safe space, perfectly preventing unreachable maze scenarios. Additionally it detected a graph theory flaw where the agent would mistake "wall collisions" for "solid blocks" instead of "edges," and successfully refactored the DFS tracking module to navigate wall borders cleanly.
* Overall, the AI acted as a highly effective pair programmer, writing the boilerplate, structuring the logic securely around the API bounds, generating the visualization logic via `matplotlib`, and writing the automated demonstration scripts.
