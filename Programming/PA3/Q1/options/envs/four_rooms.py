import numpy as np
import random
import gym
from gym import error, spaces, utils
from gym.utils import seeding

""" Four rooms. The goal is either in the 3rd room, or in a hallway adjacent to it
"""

class FourRooms(gym.Env):

    ### 1. Initializer: To initialize the state and the action space

    metadata = {'render.modes': ['human']}
    def __init__(self):

        self.rows = 11
        self.cols = 11

        self.rewards = np.zeros((self.rows,self.cols))
        self.actions = {0: [-1,0],1: [0,1],2: [0,-1], 3: [1,0]} #North, East, West, South
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(low = -10, high = 10, shape = self.rewards.shape)

        #Goals G1,G2
        self.goals = [[6,8],[8,8]]
        #Dimensions of each of the rooms
        self.room_sizes = {1: [5,5], 2: [6,5], 3: [4,5], 4: [5,5],}
        #Initialize the grid
        self.rewards[:,5] = -0.1
        self.rewards[2,5] = 0
        self.rewards[9,5] = 0
        self.rewards[5,0:5] = -0.1
        self.rewards[5,1] = 0
        self.rewards[6,5:] = -0.1
        self.rewards[6,8] = 0

        #Hallway coordinates
        self.hallways = {1: [[5,1], [2,5]], 2: [[2,5],[6,8]], 3 : [[6,8],[9,5]], 4: [[9,5],[5,1]]}

        self.viewer = None
        self.state = None

        self._seed()

    ### 2. Random seed generator

    def _seed(self, seed = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    ### 3. Step: To generate the next state and action, given the current state

    def fix_goal(self, goal):
        if goal == 'G1':
            self.rewards[6,8] = 1
            self.rewards[8,8] = 0
        if goal == 'G2':
            self.rewards[6,8] = 0
            self.rewards[8,8] = 1

    def get_start_position(self, initial_room):

        if initial_room == 1:
            start_pos = [np.random.choice(5), np.random.choice(5)]
        elif initial_room == 2:
            start_pos = [np.random.choice(6), random.randint(6,10)]
        elif initial_room == 3:
            start_pos = [random.randint(7,10), random.randint(6,10)]
        else:
            start_pos = [8,2]

        return start_pos

    def get_room_number(self,state):

        room_num = 0 #corresoponds to hallway state
        if (state[0]>=0 and state[0]<=4 and state[1]>=0 and state[1]<=4):
            room_num = 1
        if (state[0]>=0 and state[0]<=5 and state[1]>=6 and state[1]<=10):
            room_num = 2
        if (state[0]>=7 and state[0]<=10 and state[1]>=6 and state[1]<=10):
            room_num = 3
        if (state[0]>=6 and state[0]<=10 and state[1]>=0 and state[1]<=4):
            room_num = 4

        return room_num

    #Can access only if room_num != 0
    def get_hallways(self,state):
        room_num = self.get_room_number(state)
        hallways = self.hallways[room_num]
        return hallways


    def in_hallway(self,state):

        hallway1, hallway2 = self.hallways[1]
        hallway3, hallway4 = self.hallways[3]

        if state == hallway1:
            return hallway1, True
        elif state == hallway2:
            return hallway2, True
        elif state == hallway3:
            return hallway3, True
        elif state == hallway4:
            return hallway4, True
        else:
            return state, False

    def step(self,state,action,target_hallway):


        room_num = self.get_room_number(state)
        state, in_door = self.in_hallway(state)

        if  in_door:
            x = state[0] + self.actions[action][0]
            y = state[1] + self.actions[action][1]
            if self.rewards[x,y] == -0.1:
                x,y = state
                reward = -0.1
            else:
                reward = self.rewards[x,y]
            sub_goal = False
            goal = False

        else:
            x = state[0] + self.actions[action][0]
            y = state[1] + self.actions[action][1]
            hallways = self.get_hallways(state)
            #2 hallways corresponding to the room
            hallway1, hallway2 = hallways
            L = [i for i in range(0,11)]

            if ([x,y] == hallway1 or [x,y] == hallway2):
                reward = self.rewards[x,y]
                if [x,y] == target_hallway:
                    sub_goal = True
                else:
                    sub_goal = False
                if ([x,y] == self.goals[0]):
                    goal = True
                else:
                    goal = False
            #Ignoring off-grid transitions
            elif (x not in L or y not in L) :
                [x,y] = state
                reward = -0.1
                sub_goal = False
                goal = False
            elif self.rewards[x,y] == -0.1:
                [x,y] = state
                reward = -0.1
                sub_goal = False
                goal = False
            else:
                reward = self.rewards[x,y]
                if ([x,y] == self.goals[0]):
                    goal = True
                else:
                    goal = False
                sub_goal = False

        return [x,y], reward, sub_goal, goal

    ### 4. Reset: Method to reset an episode

    def reset(self, initial_room, goal):

        self.fix_goal(goal)
        start_pos = self.get_start_position(initial_room)
        return start_pos

    def render(self):

        pass
