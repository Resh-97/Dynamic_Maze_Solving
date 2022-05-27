import os
import numpy as np
import random

random.seed(2022)

flag_list = [0, 1, 2, 3, 5, 6, 7, 8]
#  [0, 1, 2
#   3,    5
#   6, 7, 8]

time_list = [0, 1, 2]

# the maze of size 201*201*2
maze_cells = np.zeros((201, 201, 2), dtype=int)

# load maze
def load_maze():
    file_path = "maze/COMP6247Maze20212022.npy"
    if not os.path.exists(file_path):
        raise ValueError("Cannot find %s" % file_path)

    else:
        global maze_cells
        maze = np.load(file_path, allow_pickle=False, fix_imports=True)
        maze_cells = np.zeros((maze.shape[0], maze.shape[1], 2), dtype=int)
        for i in range(maze.shape[0]):
            for j in range(maze.shape[1]):
                maze_cells[i][j][0] = maze[i][j]
                # load the maze, with 1 denoting an empty location and 0 denoting a wall
                maze_cells[i][j][1] = 0
                # initialized to 0 denoting no fire
    return maze

# get local 3*3 information centered at (x,y).
def get_local_maze_information(x, y):
    global maze_cells
    random_location = random.choice(flag_list)
    around = np.zeros((3, 3, 2), dtype=int)
    for i in range(maze_cells.shape[0]):
        for j in range(maze_cells.shape[1]):
            if maze_cells[i][j][1] == 0:
                pass
            else:
                maze_cells[i][j][1] = maze_cells[i][j][1] - 1  # decrement the fire time

    for i in range(3):
        for j in range(3):
            if x - 1 + i < 0 or x - 1 + i >= maze_cells.shape[0] or y - 1 + j < 0 or y - 1 + j >= maze_cells.shape[1]:
                around[i][j][0] = 0  # this cell is outside the maze, and we set it to a wall
                around[i][j][1] = 0
                continue
            around[i][j][0] = maze_cells[x - 1 + i][y - 1 + j][0]
            around[i][j][1] = maze_cells[x - 1 + i][y - 1 + j][1]
            if i == random_location // 3 and j == random_location % 3:
                if around[i][j][0] == 0: # this cell is a wall
                    continue
                ran_time = random.choice(time_list)
                around[i][j][1] = ran_time + around[i][j][1]
                maze_cells[x - 1 + i][y - 1 + j][1] = around[i][j][1]
    return around
