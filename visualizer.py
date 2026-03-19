import matplotlib.pyplot as plt
import numpy as np
from environment import MazeEnvironment

class Visualizer:
    def __init__(self, env: MazeEnvironment):
        self.env = env
        
    def show_path(self, path: list):
        maze = self.env.text_maze
        img = np.zeros((129, 129, 3), dtype=np.uint8)
        color_map = {
            'X': [0, 0, 0], 'O': [255, 255, 255], 'S': [0, 255, 0], 'G': [255, 0, 0],
            'P': [128, 128, 128], 'T': [0, 0, 255], 'C': [255, 165, 0],
            '-': [200, 200, 200], '|': [200, 200, 200]
        }
        for y in range(129):
            for x in range(129):
                char = maze[y][x]
                if char in color_map:
                    img[y, x] = color_map[char]
                else:
                    img[y, x] = [255, 255, 255]
                    
        plt.figure(figsize=(12, 12))
        plt.imshow(img, interpolation='nearest')
        
        # Plot physical path translation
        if path:
            px = [p[0]*2+1 for p in path]
            py = [p[1]*2+1 for p in path]
            plt.plot(px, py, color='blue', linewidth=2.5, alpha=0.8)
            
        plt.title("Maze Solution Path")
        plt.savefig("solution_path.png")
        print("Visualization saved to solution_path.png -> Open this to see the path!")
