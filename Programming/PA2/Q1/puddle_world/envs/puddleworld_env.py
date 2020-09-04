import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

''' Environment definition using OpenAI gym '''

class PuddleWorldEnv(gym.Env):

    ### 1. Initializer: To initialize the state and the action space

    def __init__(self):

        metadata = {'render.modes': ['human']}

        self.rows = 12
        self.cols = 12

        self.rewards = np.zeros((self.rows,self.cols))
        self.rewards[3:8,2:8] = -1
        self.rewards[4:7,3:7] = -2
        self.rewards[5:6,4:6] = -3
        self.rewards[6,5:6] = -2
        self.rewards[7,6:7] = -1
        self.rewards[8,7:8] = 0

        self.goals = [[11,0],[9,2],[7,6]]

        ## 0: Down    1:Up    2: Right    3: Left
        self.actions = {0: [0,1],1: [0,-1],2: [1,0], 3: [-1,0]}
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(low = -3, high = 10, shape = self.rewards.shape)

        self.wb = 0

        self._seed()
        self.viewer = None
        self.state = None

    ### 2. Random seed generator

    def _seed(self, seed = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    ### 3. Step: To generate the next state and action, given the current state

    def fix_goal(self,goal):

        if goal == 'A':
            x,y = self.goals[0]
            self.rewards[x,y] = 10
            self.wb = 1
            return [x,y]

        elif goal == 'B':
            x,y = self.goals[1]
            self.rewards[x,y] = 10
            self.wb = 1
            return [x,y]

        elif goal == 'C':
            x,y = self.goals[2]
            self.rewards[x,y] = 10
            self.wb = 0
            return [x,y]

        else:
            return goal


    def result_action(self, intended_action):

        prob = 0.1/3*np.ones(4)
        prob[intended_action] = 0.9

        action = np.random.choice(len([0,1,2,3]),1, p = prob)[0]

        return action


    def step(self, state, intended_action, goal):

        action = self.result_action(intended_action)

        goal = self.fix_goal(goal)

        ## Including Westerly Blowing .....

        disp = 0
        if self.wb == 1:
            disp = np.random.choice(range(2),1,[0.5,0.5])[0]


        ## Ignoring off grid transitions ......

        L = [i for i in range(0,12)]

        x = state[0] + self.actions[action][0] + disp
        y = state[1] + self.actions[action][1]

        if (x not in L or y not in L):

            x = state[0]
            y = state[1]

        reward = self.rewards[x,y]

        return [x,y], reward


    ### 4. Reset: Method to reset an episode

    def reset(self):

        start_states = [[0,5],[0,6],[0,10],[0,11]]
        self.state = start_states[np.random.randint(0, 4)]
        return self.state

    def render(self):

        pass
