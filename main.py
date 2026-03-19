import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class MazeParser:
    def __init__(self, png_path: str):
        self.png_path = png_path
        self.grid_size = 64
        self.cell_size = 16  # 1024/64 = 16
        self.pixel_size = 1024
        
        # Text representation 129x129
        self.text_maze = [[' ' for _ in range(129)] for _ in range(129)]
        
        # Fill borders
        for i in range(129):
            self.text_maze[0][i] = 'X'
            self.text_maze[128][i] = 'X'
            self.text_maze[i][0] = 'X'
            self.text_maze[i][128] = 'X'
    
    def parse(self):
        img = Image.open(self.png_path).convert('RGB')
        if img.size != (self.pixel_size, self.pixel_size):
            img = img.resize((self.pixel_size, self.pixel_size), Image.Resampling.NEAREST)
        pixels = np.array(img)
        
        for cell_y in range(self.grid_size):
            for cell_x in range(self.grid_size):
                self._parse_cell(pixels, cell_y, cell_x)
        
        return self.text_maze
    
    def _parse_cell(self, pixels, cell_y, cell_x):
        text_y = cell_y * 2 + 1
        text_x = cell_x * 2 + 1
        
        pixel_y = cell_y * self.cell_size
        pixel_x = cell_x * self.cell_size
        center_y = pixel_y + self.cell_size // 2
        center_x = pixel_x + self.cell_size // 2
        r, g, b = pixels[center_y, center_x]
        
        self.text_maze[text_y][text_x] = self._get_cell_char(r, g, b)
        
        # Right connection
        if cell_x < self.grid_size - 1:
            if self._check_vertical_wall(pixels, pixel_x + self.cell_size, pixel_y, pixel_y + self.cell_size):
                self.text_maze[text_y][text_x + 1] = 'X'
            else:
                self.text_maze[text_y][text_x + 1] = '-'
        
        # Down connection
        if cell_y < self.grid_size - 1:
            if self._check_horizontal_wall(pixels, pixel_y + self.cell_size, pixel_x, pixel_x + self.cell_size):
                self.text_maze[text_y + 1][text_x] = 'X'
            else:
                self.text_maze[text_y + 1][text_x] = '|'
        
        # Diagonal corner
        if cell_x < self.grid_size - 1 and cell_y < self.grid_size - 1:
            self.text_maze[text_y + 1][text_x + 1] = 'X'
    
    def _get_cell_char(self, r, g, b):
        if r > 200 and g < 100 and b < 100: return 'S'
        if r < 100 and g > 200 and b < 100: return 'G'
        if abs(r-g)<30 and abs(g-b)<30 and 100<r<180: return 'P'
        if r<100 and g<100 and b>200: return 'T'
        if r>200 and g>100 and b<50: return 'C'
        if r<50 and g<50 and b<50: return 'X'
        return 'O'
    
    # New robust vertical wall detection
    def _check_vertical_wall(self, pixels, x, y_start, y_end):
        wall_count = 0
        total = 0
        thickness = 3  # sample a 3-pixel wide vertical band
        for dx in range(-thickness//2, thickness//2 + 1):
            bx = min(max(0, x + dx), self.pixel_size - 1)
            for y in range(y_start, y_end, 2):
                r, g, b = pixels[y, bx]
                if r < 80 and g < 80 and b < 80:
                    wall_count += 1
                total += 1
        return (wall_count / total) > 0.3 if total > 0 else True
    
    # New robust horizontal wall detection
    def _check_horizontal_wall(self, pixels, y, x_start, x_end):
        wall_count = 0
        total = 0
        thickness = 3  # sample a 3-pixel tall horizontal band
        for dy in range(-thickness//2, thickness//2 + 1):
            by = min(max(0, y + dy), self.pixel_size - 1)
            for x in range(x_start, x_end, 2):
                r, g, b = pixels[by, x]
                if r < 80 and g < 80 and b < 80:
                    wall_count += 1
                total += 1
        return (wall_count / total) > 0.3 if total > 0 else True
    
    def print_maze(self):
        for row in self.text_maze:
            print(''.join(row))
    
    def save_to_file(self, filename):
        with open(filename, 'w') as f:
            for row in self.text_maze:
                f.write(''.join(row) + '\n')
    
    def visualize(self):
        color_map = {
            'X':[0,0,0],'O':[255,255,255],'S':[255,0,0],'G':[0,255,0],
            'P':[128,128,128],'T':[0,0,255],'C':[255,165,0],
            '-':[200,200,200],'|':[200,200,200]
        }
        img = np.zeros((129,129,3),dtype=np.uint8)
        for y in range(129):
            for x in range(129):
                img[y,x] = color_map.get(self.text_maze[y][x],[255,0,255])
        plt.figure(figsize=(12,12))
        plt.imshow(img,interpolation='nearest')
        plt.grid(True,color='yellow',alpha=0.2,linewidth=0.5)
        plt.show()

# Usage
if __name__ == "__main__":
    parser = MazeParser("MAZE_0.png")
    parser.parse()
    parser.print_maze()
    parser.visualize()
    parser.save_to_file("maze_64x64.txt")