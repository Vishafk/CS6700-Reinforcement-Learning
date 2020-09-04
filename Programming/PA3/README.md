# Reinforcement Learning - HRL, DQN

Implementation of Heirarchical RL and DQN using OpenAI Gym

## Algorithms

1. SMDP QLearning
2. Intra-Option QLearning
3. Deep Q-Network (DQN)

## Description

SMDP QLearning and Intra-Option QLearning are performed on a discrete grid world with hallways options

A DQN is trained to solve the 'CartPole' task

Evolution of reward value, number of steps taken for convergence and the learnt Q values are used to evaluate performance

## USAGE

First, install and create the environments in your local system using OpenAI Gym and then run the algorithms using the given IPython Jupyter Notebook file run.ipynb

For the CartPole task, directly run the kernels in the Jupyter Notebook dqn.ipynb

## Dependencies

* gym
* tensorflow
* IPython3
* numpy
* matplotlib

## References

* [Reinforcement Learning : An Introduction](http://incompleteideas.net/book/RLbook2018.pdf) by Richard and Sutton
* [Between MDPs and semi-MDPs:](https://people.cs.umass.edu/~barto/courses/cs687/Sutton-Precup-Singh-AIJ99.pdf) A framework for temporal abstraction in reinforcement learning
* [Intra-Option Learning about Temporally Abstract actions](https://web.eecs.umich.edu/~baveja/Papers/ICML98_SPS.pdf)
* [CartPole Environment OpenAI Gym](https://gym.openai.com/envs/CartPole-v0/)
