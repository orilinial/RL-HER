# Deep RL - Hindsight Experience Replay (HER)
This repository holds deep RL solutions for solving the bit flipping enviroment using the hindsight experience replay. </br>
View the original paper <a href="https://arxiv.org/abs/1707.01495">here</a>.

###### Bit-Flip Environment
In this environment, we are given with a starting state with is a binary vector of size n, and a goal state of size n. </br>
In each action, the user can flip one of the bits in the current state. For each step, the user gets a reward '-1', </br>
and for a step which makes the current state equal to the goal, the user gets a reward '0'. </br>
The environment is written in the `bit_flip_env.py` file.

###### Bit-Flip Dynamic Environment
To check the ability of HER to deal with dynamic environments, we added this option to the bit flipping domain.</br>
This means that with every step the user makes, with probability 0.3, one of the goal's bits would flip, </br>
making it harder to predict. The goal's flipped bit is chosen with uniform probability. </br>

###### Hindsight Experience Replay (HER)
The algorithm, described in details <a href="https://arxiv.org/abs/1707.01495">here</a> by Andrychowicz et al. can deal with sparse binary rewards (as we get in the bit flipping domain. </br>
The problem with sparse rewards, is that for very large state spaces, we might never get a succesful episode, making it very hard to learn. </br>
In this algorithm, we create new "fake" episodes from unsuccesful ones, by chaging their original goal to one of the states they actually reached. </br>
This way, we add successes to the experience replay buffer, and can learn from them. It is basically the same as learning from mistakes.

##### Hindsight Experience Replay with Dynamical goals (DHER)
The concept here is very similar to HER, and is described <a href="https://openreview.net/forum?id=Byf5-30qFX">here</a> by Fang et al. </br>
This algorithm takes also into account that the goal made some trasitions over time, and uses its trajectory to learn how to reach it.

###### Scripts Usage:
All the files below have arguments which can be changed (but all set by default to our choice of parameters). </br>
To see all arguments for each script run: `<SCRIPT NAME>.py --help` </br>
Example for running a script: `python main.py`

### Train scripts:
To train the model that solves the bit flipping environment, run the following scripts: `main.py` </br>.
Note that the argument `--state-size <NUMBER>` is neccesary, in order to see the effect of the different sizes on the model.</br>
Adding the argument `--HER` or `--DHER` would use the respective algorithms. </br>
Adding the argument `--dynamic` would use the dynamical mode of the environment.
The models architecture is specified in: `dqn.py`

### Evaluation scripts:
To test the models run the following scripts: `evaluate_model.py` with the relevant `--state-size` argument. </br>
We added a trained model in the `bit_flip_model.pkl` file, with the size n=10.

### Results


### Example of the evaluation run


