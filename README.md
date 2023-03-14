# Snake AI

## About this project
This project was part of a main project, during a course in Introduction to AI.

## A snake ai using deep reinforced learning
The game uses reinforced learning technique to play the game of snake

## Reinforced Learning
Reinforcement learning is an area of machine learning concerned with how software agents ought to take actions in an environment in order to maximize the notion of cumulative reward.
The goal is to maximize reward by finding the best action to take given the current state of the environment.
The agent is not told which actions are correct, only which ones yield the most reward.
The agent must discover which actions yield the most reward by trial and error, and this trial-and-error search for the best action is called reinforcement learning.
Reinforcement learning is one of three basic machine learning paradigms, alongside supervised learning and unsupervised learning.

## Folder structure

*  ` - [dep] -- Contains the requirements needed to run the game`
*  ` - [scripts] -- Contains the necessary scripts to install and run the application, using make`
*  ` - [src] -- Contains the game files`
*  ` - [Makefile] -- Makefile`

There are necessary nix files that can be used if system applicable

## Run the game
1.  `make build` -- This will build the necessary requirements
2.  `make run` -- This will run the game, follow instructions in game menu
3.  `make clean` -- clean out the dependency

## After each training
*  When the training is done, a new weight will be written to `./src/weights`
*  and a plot for the training is created and can be found in `./src/plots`

### Features
-  [x] Game menu
-  [x] Single layered network
-  [x] Deep network
-  [ ] Convolutional network (Currently not working)
-  [x] Easy selector for number of games
-  [ ] Model switcher (easy switch between models)
-  [ ] Additional snakes, using different approaches

As of now, switching between different models, have to be done manualy.
Go to `./src/agent.py` and on line 25, change from linear to e.g deep.
The Convolutional network is not implemented, only options are deep and linear.

## Multiple snakes
The idea is to implement multiple snakes, using different search algorithm, avoiding the main snake

## Debug
*  When pushing Model switcher (easy switch between model) it terminates the game, due to not yet implement function.
*  Game number increments in perpetuity, after training is done.
