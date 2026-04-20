from environment import MazeEnvironment, Action

def demo():
    print("Loading Maze with hazards (MAZE 1)...")
    env = MazeEnvironment('maze_1.txt')
    
    if env.pits:
        px, py = env.pits[0]
        print("\n--- Demonstrating Death Pit ---")
        # Ensure adjacent cell is open
        env._set_cell(px, py-1, 'O')
        env.start_pos = (px, py)
        env.reset()
        print(f"Agent starts adjacent to pit. Moving DOWN into Pit at {(px, py)}")
        env.current_pos = (px, py-1)
        res = env.step([Action.MOVE_DOWN])
        print(f"Action: MOVE_DOWN")
        print(f"Result -> is_dead: {res.is_dead}, deaths count: {env.deaths}")
        print(f"Agent respawned at: {res.current_position}")
        
    if env.teleporters:
        tx, ty = env.teleporters[0]
        print("\n--- Demonstrating Teleport ---")
        dest = env.teleport_map[(tx, ty)]
        env._set_cell(tx, ty-1, 'O')
        env.reset()
        env.current_pos = (tx, ty-1)
        print(f"Agent adjacent to Teleport at {(tx,ty)}. Expected Destination: {dest}")
        res = env.step([Action.MOVE_DOWN])
        print(f"Action: MOVE_DOWN")
        print(f"Result -> teleported: {res.teleported}, current_position: {res.current_position}")
        
    if env.confusion_pads:
        cx, cy = env.confusion_pads[0]
        print("\n--- Demonstrating Confusion Pad ---")
        env._set_cell(cx, cy-1, 'O')
        env._set_cell(cx-1, cy, 'O')
        env.reset()
        env.current_pos = (cx, cy-1)
        print(f"Agent adjacent to Confusion Pad at {(cx,cy)}.")
        print("Submitting 2 actions: [MOVE_DOWN, MOVE_DOWN]")
        print("Expected behavior: First move steps on pad, applying confusion mid-turn.")
        print("The second MOVE_DOWN should be inverted into a MOVE_UP!")
        res = env.step([Action.MOVE_DOWN, Action.MOVE_DOWN])
        print(f"Result after turn: Confused status: {res.is_confused}, End position: {res.current_position}")
        
        print("\nNext turn: Submitting [MOVE_RIGHT]")
        print("Expected behavior: Agent actually moves LEFT.")
        prev_x, prev_y = res.current_position
        res2 = env.step([Action.MOVE_RIGHT])
        new_x, new_y = res2.current_position
        actual_move = ""
        if new_x < prev_x: actual_move = "LEFT"
        elif new_x > prev_x: actual_move = "RIGHT"
        elif new_y < prev_y: actual_move = "UP"
        elif new_y > prev_y: actual_move = "DOWN"
        else: actual_move = "BLOCKED BY WALL"
        print(f"Agent actual movement on the grid: {actual_move} (Position went from {prev_x, prev_y} to {new_x, new_y})")

if __name__ == "__main__":
    demo()
