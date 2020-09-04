import gym
import numpy as np
from matplotlib import pyplot as plt
import multiprocessing as mp
import options
from tqdm import tqdm
from random import randint, uniform


class SMDPQ:

    # epsilon-greedy way of picking options
    def pick_option(self, state, Q, epsilon):
        rand = uniform(0,1)
        if rand < epsilon:
            option_index = np.random.choice(6)
        else:
            option_index = np.argmax(Q[state[0], state[1],:])

        return option_index

    #Identify the target hallway given an option
    def get_target_hallway(self, state, option, env):

        option_index = option - 4
        hallway1, hallway2 = env.hallways[1]
        hallway3, hallway4 = env.hallways[3]
        if state == hallway1 and option_index == 0:
            target = env.hallways[4][0]
        elif state == hallway1 and option_index == 1:
            target = env.hallways[1][1]
        elif state == hallway2 and option_index == 0:
            target = env.hallways[1][0]
        elif state == hallway2 and option_index == 1:
            target = env.hallways[2][1]
        elif state == hallway3 and option_index == 0:
            target = env.hallways[2][0]
        elif state == hallway3 and option_index == 1:
            target = env.hallways[3][1]
        elif state == hallway4 and option_index == 0:
            target = env.hallways[3][0]
        elif state == hallway4 and option_index == 1:
            target = env.hallways[4][1]
        else:
            room_num = env.get_room_number(state)
            target = env.hallways[room_num][option_index]
        return target

    #Include stochasticity for primitive actions
    def result_action(self, intended_action):

        prob = 1/9*np.ones(4)
        prob[intended_action] = 2/3

        action = np.random.choice(len([0,1,2,3]),1, p = prob)[0]

        return action

    def execute_option(self, state, alpha, gamma, epsilon, option, target_hallway, intra, Q, env):

        steps = 0
        tot_reward = 0
        g = 1.0
        #Primitive actions
        if option < 4:

            action = self.result_action(option)
            next_state, reward, sub_goal, goal = env.step(state, action, target_hallway)
            steps += 1
            tot_reward = reward
            if intra:
                Q[state[0], state[1], option] = (1.0 - alpha) * Q[state[0], state[1], option] + alpha * (reward + gamma * max(Q[next_state[0], next_state[1],:]))
        #Hallway options
        else:
            while state != target_hallway:
                action = self.decode_option(state, target_hallway, env)
                next_state, reward, sub_goal, goal = env.step(state, action, target_hallway)
                next_option = self.pick_option(next_state, Q, epsilon)

                #Intra-Option update
                if intra:
                    if not sub_goal:
                        Q[state[0], state[1], option] = (1.0 - alpha) * Q[state[0], state[1], option] + alpha * (reward + gamma * (Q[next_state[0], next_state[1],next_option]))
                    else:
                        Q[state[0], state[1], option] = (1.0 - alpha) * Q[state[0], state[1], option] + alpha * (reward + gamma * max(Q[next_state[0], next_state[1], :]))

                state = next_state
                tot_reward += reward*g
                g*= gamma
                steps += 1

                if (sub_goal or goal):
                    break


        return next_state, tot_reward, steps, sub_goal, goal, Q

    # Returns best primitve action to reach hallway by minimizing manhatten dist
    def decode_option(self, state, target_hallway, env):

        x1,y1 = state
        x2,y2 = target_hallway

        state, flag = env.in_hallway(state)
        if x2>x1:
            action1 = 3
        elif x1>x2:
            action1 = 0
        if y2>y1:
            action2 = 1
        elif y1>y2:
            action2 = 2

        if flag:

            disp = env.rewards[x1 + env.actions[action1][0], y1]
            if (disp<0):
                action = action2
            else:
                action = action1

        else:
            if x1 == x2:
                action = action2
            elif y1==y2:
                action = action1
            else:
                if (abs(x2-x1) > abs(y2-y1) and (env.rewards[state[0] + env.actions[action1][0],y1])>=0):
                    action = action1
                elif (abs(x2-x1) < abs(y2-y1) and (env.rewards[x1,state[1] + env.actions[action2][1]])>=0):
                    action = action2
                else:
                    if env.rewards[state[0]+env.actions[action1][0],y1] >=0:
                        action = action1
                    else:
                        action = action2

        return action


    # Update rule in case of QLearning
    def update_rule(self, state, option, reward, k, next_state, Q, alpha, gamma):

        Q[state[0], state[1], option] = (1.0 - alpha) * Q[state[0], state[1], option] + alpha * (reward + (gamma**k) * max(Q[next_state[0], next_state[1],:]))

        return Q

    def episode_run(self, gamma, alpha, epsilon, initial_room, goal, num_episodes, intra, env):

        rewards = []
        num_steps = []

        # Random initialization
        Q = np.zeros((env.observation_space.shape[0], env.observation_space.shape[1], env.action_space.n + 2))
        for episode in range(num_episodes):

            #Initialize the start state
            state = env.reset(initial_room, goal)

            ep_reward = 0
            num_step = 0

            while 1:

                #Pick an option and exucte it
                option = self.pick_option(state, Q, epsilon)
                if option == 4 or option == 5:
                    target_hallway = self.get_target_hallway(state, option, env)
                else:
                    target_hallway = None

                #Compute reward r(s,o) and update Q(s,o)
                next_state, reward, steps, sub_goal, goal, Q = self.execute_option(state, alpha, gamma, epsilon, option, target_hallway, intra, Q, env)
                if not intra:
                    # Change Q based on the update rule
                    Q = self.update_rule(state, option, reward, steps, next_state, Q, alpha, gamma)
                state = next_state
                ep_reward += reward
                num_step += steps

                if goal:
                    break

            rewards.append(ep_reward)
            num_steps.append(num_step)

        return rewards, num_steps, Q
