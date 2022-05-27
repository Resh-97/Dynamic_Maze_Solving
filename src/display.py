from maze.read_maze import load_maze
import pygame
import sys

''' Code has been adapted from Pygame documentation and examples'''

'''Setting screen size'''
SCREENSIZE = W, H = 201, 201
mazeWH = 201
origin = (1,1)
lw = 1

"""
RED - signifies a fire
GREEN - indicates the actor's position
BLUE - indicates goal sate
"""
GREY = (140,140,140)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)

class Display:
    def __init__(self):
        self.step_cntr = 0
        self.cntr = 0

        self.maze = load_maze()
        self.maze = self.maze.T
        self.shape = 201

        pygame.init()
        self.font = pygame.font.SysFont('Arial', 16)

        self.surface = pygame.display.set_mode(SCREENSIZE)
        self.actor = (1,1)

    def drawSquareCell(self, x, y, dimX, dimY, col=(0, 0, 0)):
        pygame.draw.rect(
            self.surface, col,
            (x, y, dimX, dimY)
        )

    ''' Function to draw the grid border'''
    def drawSquareGrid(self, origin, gridWH):
        CONTAINER_WIDTH_HEIGHT = gridWH
        cont_x, cont_y = origin

        pygame.draw.line(
            self.surface, BLACK,
            (cont_x, cont_y),
            (CONTAINER_WIDTH_HEIGHT + cont_x, cont_y), lw)

        pygame.draw.line(
            self.surface, BLACK,
            (cont_x, CONTAINER_WIDTH_HEIGHT + cont_y),
            (CONTAINER_WIDTH_HEIGHT + cont_x,
             CONTAINER_WIDTH_HEIGHT + cont_y), lw)

        pygame.draw.line(
            self.surface, BLACK,
            (cont_x, cont_y),
            (cont_x, cont_y + CONTAINER_WIDTH_HEIGHT), lw)

        pygame.draw.line(
            self.surface, BLACK,
            (CONTAINER_WIDTH_HEIGHT + cont_x, cont_y),
            (CONTAINER_WIDTH_HEIGHT + cont_x,
             CONTAINER_WIDTH_HEIGHT + cont_y), lw)


    def placeCells(self):
        cellBorder = 0
        celldimX = celldimY = (mazeWH / self.shape)

        for rows in range(201):
            for cols in range(201):
                if (self.maze[rows][cols] == 0):
                    self.drawSquareCell(
                        origin[0] + (celldimY * rows)
                        + cellBorder + lw / 2,
                        origin[1] + (celldimX * cols)
                        + cellBorder + lw / 2,
                        celldimX, celldimY, col=BLACK)
                if cols == 1 and rows == 1:
                    self.drawSquareCell(
                        origin[0] + (celldimY * rows)
                        + cellBorder + lw / 2,
                        origin[1] + (celldimX * cols)
                        + cellBorder + lw / 2,
                        celldimX, celldimY, col=BLUE)

    '''Function to display the trajectory of the actor in the maze'''
    def step(self, visible, idx, path, acts, action, score, reward, tep_cntr, wall, visit, obs, name):
        self.get_event()
        self.set_visible(visible, idx, path)
        self.surface.fill(GREY)
        self.drawSquareGrid(origin, mazeWH)
        self.placeCells()
        self.draw_visible(acts, action,  score, reward, tep_cntr, wall, visit, obs)
        pygame.image.save(self.surface, "screenshot_"+ name+ ".jpeg")
        pygame.display.update()
        self.step_cntr += 1

    def set_visible(self, visible, idx, path):
        self.vis = visible   #Environement copy
        self.actor = idx   ##Actor positions
        self.path = path

    def draw_visible(self, acts, action, score, reward, step_cntr,wall, visit, obs):

        celldimX = celldimY = (mazeWH / self.shape)
        for s in self.path:
            self.drawSquareCell(
                origin[0] + (celldimX * s[0]) + lw / 2,
                origin[1] + (celldimY * s[1]) + lw / 2,
                celldimX, celldimY, col=RED)

        self.drawSquareCell(
            origin[0] + (celldimX * self.actor[0]) + lw / 2,
            origin[1] + (celldimY * self.actor[1]) + lw / 2,
            celldimX, celldimY, col=GREEN)

        cell_size = 12
        for row, obs_row in enumerate(obs):
            for col, obs_i in enumerate(obs_row):
                c = self.actor[0] + (col - 1)
                r = self.actor[1] + (row - 1)
                self.drawSquareCell(
                    1100 + (cell_size*(col - 1)),
                    350 + (cell_size*(row - 1)),
                    cell_size, cell_size, col=obs_i)

    ''' Exceuted when the pygame window is closed'''
    def get_event(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
