from maze.read_maze import get_local_maze_information
import numpy as np
global rewards
global actions

'''Define the action space'''

actions = {"0": {"id":'stay',
                    "move":(0,0)},
              "1": {"id":'up',
                    "move":(0,-1)},
              "2": {"id":'left',
                    "move":(-1,0)},
              "3": {"id":"down",
                    "move":(0,1)},
              "4": {"id":'right',
                    "move":(1,0)}
              }
''' Defines the reward function'''

rewards = {"forward": +1.,
            "visited":-.05,
            "wall":-.1,
            "away":+.8,
            "stay":-0.01
              }

class Environment:
    def __init__(self):

        '''Initialise counters, actor location and observation map'''
        self.steps = 0
        self.walls = 0
        self.n_stay = 0
        self.visits = 0
        self.actor_position = (1, 1)
        self.actorpath = [self.actor_position]
        self.observation_map = [[-1 for i in range(200)] for j in range(200)]
        self.observation_size = 10
        self.not_observed = 0
        self.observation = self.observe_environment
        o = self.observation[:-3]  # set up empty state
        self.observations = [o[(i) * int(len(o) / 3):(i + 1) * int(len(o) / 3)] for i in range(3)]
        self.obs2D = []


    '''Function to re-initialise the value before every episode'''
    @property
    def re_initialise(self):
        self.steps = 0
        self.walls = 0
        self.n_stay = 0
        self.visits = 0

        self.actor_position = (1, 1)
        self.actorpath = [self.actor_position]

        self.observation_size = 10
        self.observation_map = [[-1 for i in range(200)] for j in range(200)]
        self.not_observed = 0
        self.observation = self.observe_environment
        o = self.observation[:-3]
        self.observations = [o[(i) * int(len(o) / 3):(i+1) * int(len(o) / 3)] for i in range(3)]

        return self.observation


    ''' Create a 2D map render the local state of the maze and create screenshopts later'''
    def observation_2dlarge(self, l1, name):
        obsv = [[[0, 0, 0] for j in range(self.observation_size)] for i in range(self.observation_size*3)]
        for i, l in enumerate(l1):
            c, d = i % self.observation_size, int(i / self.observation_size)
            if l == 1:
                obsv[d][c] = [0, 255, 0]
            elif l == 2:
                obsv[d][c] = [255, 255, 255]
            elif l == 0:
                obsv[d][c] = [0, 0, 0]
            else:
                obsv[d][c] = [0, 140, 50]

        return obsv

    ''' Create a 2D map render the local state of the maze and create screenshopts later'''
    def observation_2d(self, l1, name):
        obsv = [[[0, 0, 0] for j in range(self.observation_size)] for i in range(self.observation_size)]
        for i, l in enumerate(l1):
            c, d = i % self.observation_size, int(i / self.observation_size)
            if l == 1:
                obsv[d][c] = [0, 255, 0]
            elif l == 2:
                obsv[d][c] = [255, 255, 255]
            elif l == 0:
                obsv[d][c] = [0, 0, 0]
            elif l == -1:
                obsv[d][c] = [0, 140, 50]
            else:
                obsv[d][c] = [0, 50, 50]

        return obsv

    @property
    def observe_environment(self):
        x, y = self.actor_position
        pos = [(x + j - 1, y + i -1) for i in range(3) for j in range(3)]

        loc = get_local_maze_information(y, x)
        self.loc = loc

        '''Fetch the local binary values and create an observation map'''
        for a, b in pos:
            i, j = a-x+1, b-y+1 # (0,0) (0,1)(0,2), (1,0) ... (2,2)
            if loc[j][i][0] == 0:
                self.observation_map[b][a] = 0
            else:
                self.observation_map[b][a] = 2

        self.observation_map[y][x] = 1

        if self.actor_position not in self.actorpath:
            self.visits = 0
            self.actorpath.append(self.actor_position)
        else:
            self.visits += 1

        c = int(self.observation_size / 2)

        a_1, a_2 = self.actor_position[0] - c, self.actor_position[0] + c
        b_1, b_2 = self.actor_position[1] - c, self.actor_position[1] + c

        '''Fetches 21 x 21 local observation'''
        l1 = []
        for j in range(b_1, b_2):
            for i in range(a_1, a_2):

                if j < 0 or i < 0:
                    l1.append(-2)
                else:
                    l1.append(self.observation_map[j][i])

        if self.not_observed != 0:
            l2 = self.observations[0].copy()

            self.observations[0] = l1

            # only change prior observation when a change has occured
            if l2 != l1:
                self.observations.append(l2)
                self.observations.pop(1)
                # self.prior_observation = l2


            l = self.observations[0] + self.observations[2] + self.observations[1]

            self.obs2D = self.observation_2dlarge(l, '2')

            self.observation = l + [self.actor_position[0], self.actor_position[1], len(self.actorpath)]
            return self.observation

        else:
            l2 = l1.copy()
            self.not_observed += 1
            l = l1 + l2 + l2
            self.obs2D = self.observation_2dlarge(l, '2')

            o = l + [self.actor_position[0], self.actor_position[1], len(self.actorpath)]
            return o

    '''Get actor position'''
    @property
    def get_actor_position(self):
        return self.actor_position

    ''' For Evaluation Phase:Function calculates reward and location'''
    def step_test(self, action, score):

        '''Fetch action and reward distionary'''
        global actions
        global rewards

        self.steps += 1
        act_key = str(action)

        '''new location'''
        x_inc, y_inc = actions[act_key]['move']
        '''current location'''
        x, y = self.actor_position
        '''updated location'''
        x_loc, y_loc = (1 + x_inc, 1 + y_inc)

        '''location in observation map'''
        obsv_mat = self.loc

        '''Case: Stay. Penalise. Update counter'''
        if actions[act_key]['id'] == 'stay':
            self.n_stay += 1
            return self.observe_environment, rewards['stay'], False, {}

        '''Case: Wall. Penalise. Update counter.'''
        if obsv_mat[y_loc][x_loc][0] == 0:
            self.walls += 1
            return self.observe_environment, rewards['wall'], False, {}

        self.actor_position = new_pos = (x + x_inc, y + y_inc)

        '''Case: Goal Reached. Hooray!'''
        if new_pos == (36, 13):
            print('Goal Reached')
            return self.observation, 10000., True, {}

        '''Case: Location previously visited; Then update visit counter'''
        if self.actor_position in self.actorpath:
            return self.observe_environment, rewards['visited'], False, {}

        '''Case: Moving Towards the goal'''
        if x_inc > 0 or y_inc > 0:
            return self.observe_environment, rewards['forward'], False, {}



        '''Case: Moving away from objective'''
        return self.observe_environment, rewards['away']*(len(self.actorpath)/10), False, {}


    ''' For Training Phase:Function calculates reward and location'''
    def step(self, action, score):

        '''Fetch action and reward distionary'''
        global actions
        global rewards

        self.steps += 1
        act_key = str(action)

        '''new location'''
        x_inc, y_inc = actions[act_key]['move']
        '''current location'''
        x, y = self.actor_position
        '''updated location'''
        x_loc, y_loc = (1 + x_inc, 1 + y_inc)

        '''location in observation map'''
        obsv_mat = self.loc


        '''Case: Spent too long in the maze. Termintate!!!'''
        if self.steps > len(self.actorpath)*4 or self.steps > 4000:
            print('Visit timeout')
            return self.observe_environment, -0., True, {}

        '''Case: Extreme contraint used only for training. Termintate!!!'''
        if self.visits > 1 or self.walls > 1:
            print('Banging your head too many times on the wall')
            return self.observe_environment, -1., True, {}

        '''Case: Stay. Penalise. Update counter'''
        if actions[act_key]['id'] == 'stay':
            self.n_stay += 1
            return self.observe_environment, rewards['stay'], False, {}

        '''Case: Wall. Penalise. Update counter.'''
        if obsv_mat[y_loc][x_loc][0] == 0:
            self.walls += 1
            return self.observe_environment, rewards['wall'], False, {}

        self.actor_position = new_pos = (x + x_inc, y + y_inc)
        '''Case: Location previously visited; Then update visit counter'''
        if self.actor_position in self.actorpath:
            return self.observe_environment, rewards['visited'], False, {}

        '''Case: Moving Towards the goal'''
        if x_inc > 0 or y_inc > 0:
            return self.observe_environment, rewards['forward'], False, {}

        '''Case: Goal Reached. Hooray!'''
        if new_pos == (36, 13):
            print('Goal Reached')
            return self.observation, 10000., True, {}

        '''Case: Moving away from objective'''
        return self.observe_environment, rewards['away']*(len(self.actorpath)/10), False, {}
