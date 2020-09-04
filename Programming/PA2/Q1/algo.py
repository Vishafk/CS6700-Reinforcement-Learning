import gym
import numpy as np
from matplotlib import pyplot as plt
import multiprocessing as mp
import warnings
import puddle_world
from tqdm import tqdm
from random import randint, uniform
warnings.simplefilter('ignore')

''' QLearning and SARSA algorithms'''

class QLearning:

    # Update rule in case of QLearning
    def update_rule(self, curr_state, curr_action, reward, next_state, Q, alpha, gamma):

        Q[curr_state[0], curr_state[1], curr_action] = (1.0 - alpha) * Q[curr_state[0], curr_state[1], curr_action] + alpha * (reward + gamma * max(Q[next_state[0], next_state[1],:]))

        return Q

   # epsilon-greedy way of picking actions
    def pick_action(self, state, Q, epsilon):
        rand = uniform(0,1)
        if rand < epsilon:
            action_index = np.random.choice(4)
        else:
            action_index = np.argmax(Q[state[0], state[1], :])

        return action_index

    # Sample trajectories and run QLearning
    def episode_run(self, gamma, alpha, epsilon, num_episodes, env, goal):

        rewards = []
        num_steps = []

        # Target
        goal = env.fix_goal(goal)

        # Random initialization
        Q = np.random.rand(env.observation_space.shape[0], env.observation_space.shape[1], env.action_space.n)

        for episode in range(num_episodes):

            # Initialize the start state
            curr_state = env.reset()

            ep_reward = 0
            num_step = 0

            while 1:

                # pick an action and apply it to go to next state
                curr_action = self.pick_action(curr_state, Q, epsilon)
                next_state, reward = env.step(curr_state, curr_action, goal)

                # Change Q based on the update rule
                Q = self.update_rule(curr_state, curr_action, reward, next_state, Q, alpha, gamma)

                curr_state = next_state

                num_step += 1
                ep_reward += reward

                if curr_state == goal:
                    break

            rewards.append(ep_reward)
            num_steps.append(num_step)

        return rewards, num_steps, Q


class SARSA:

    # Update rule in case of SARSA
    def update(self, curr_state, curr_action, reward, next_state, next_action, Q, E, alpha, gamma, lambd):

        if not (lambd):

            Q[curr_state[0],curr_state[1],curr_action] = (1.0 - alpha) * Q[curr_state[0],curr_state[1],curr_action] + alpha * (reward + gamma*Q[next_state[0],next_state[1],next_action])

        else:
            delta_t = reward + gamma * Q[next_state[0],next_state[1],next_action] - Q[curr_state[0], curr_state[1],curr_action]
            E[curr_state[0],curr_state[1],curr_action] += 1.0
            Q += alpha * delta_t * E
            E *= gamma * lambd

        return Q,E

    # epsilon-greedy way of picking actions
    def pick_action(self,epsilon, state, Q, env):

        # The exploration part
        if np.random.uniform(0,1) < epsilon:
            action = np.random.choice(4)
        # The expoition part
        else:
            action = np.argmax(Q[state[0],state[1],:])
        return action

    # Sample trajectories and run SARSA
    def episode_run(self, gamma, alpha, lambd, epsilon, num_episodes, env, goal):

        # Target
        goal_pos = env.fix_goal(goal)

        # Random initialization
        Q = np.random.rand(env.observation_space.shape[0], env.observation_space.shape[1], env.action_space.n)

        rewards = []
        num_steps = []

        for episode in range(num_episodes):

            # Eligibility parameter initialization
            E = np.zeros((env.observation_space.shape[0], env.observation_space.shape[1], env.action_space.n))

            ep_reward = 0
            num_step = 1

            curr_state = env.reset()

            # Pick an action
            curr_action = self.pick_action(epsilon, curr_state, Q, env)

            while 1:

                # Sample states based on the action picked
                next_state, reward = env.step(curr_state, curr_action, goal)

                next_action = self.pick_action(epsilon, next_state, Q, env)

                # Update Q and E based on the update rule
                Q,E = self.update(curr_state, curr_action, reward, next_state, next_action, Q, E, alpha, gamma, lambd)

                curr_state = next_state
                curr_action = next_action

                num_step +=1
                ep_reward += reward

                if curr_state == goal_pos:
                    break

            rewards.append(ep_reward)
            num_steps.append(num_step)

        return rewards, num_steps, Q
