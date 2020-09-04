# Bandit Problem - Reinforcement Learning

Implementation of multi-armed bandits using IPython Jupyter Notebook

## Algorithms

1. Epsilon-greedy
2. Softmax action selection
3. UCB1 - Upper confidence bound
4. Median Elimination algorithm (PAC bounds)

## Description

Experiments are conducted on a 10-arm bandit tested. The true value of each of the arm is sampled from a normal distribution of zero mean and unit variance, and then the actual rewards are selected according to a normal distribution with the true value of the arm as the mean and variance 1. 

Each of the above 4 algorithms is run on the 10 arm bandit test bed and their performance is compared. Also, the effects of varying hyperparameters on a particular method are analyzed. Finally, the optimal parameter settings are used to compare all the algorithms with a significantly large testbed size (1000 arms)

## Dependencies

* IPython3
* numpy
* matplotlib

## References

* [Reinforcement Learning : An Introduction](http://incompleteideas.net/book/RLbook2018.pdf) by Richard and Sutton
