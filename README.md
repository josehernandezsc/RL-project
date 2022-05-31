# RL-project
This repository contains several implementations of various Reinforcement Learning (RL) algorithms.
First, an implementation to solve a simple decision-making problem using a model-based MDP algorithm is used, iterating on the Bellman equations using an iterative policy method. Then, model free algorithms are explored. First visit Monte Carlo and tabular Q-learning methods are implemented to solve the Taxi v3 environment. Moving into Deep RL, a Deep Q-Learning algorithm is implemented to contro a lunar probe to smoothly land on the moon. To control for this, a decaying alpha method was implemented plus a decaying epsilon exploration constant. Hard updates on the baseine (bootstrapped estimate) are taken.

A deep policy gradient algorithm is implemented. Besides the decaying constant, an increasing tau algorithm based on simulated annealing and tanh function is implemented to guide convergence, this proved to be an innovative method which motivated further research in tau controlled increment. It was tested on a bipedal robot which was taught (learned) how to run without falling.

Finally a multi-agent RL algorithm was implemented to solve a matrix game problem.
