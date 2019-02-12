import numpy as np
import random

class BitFlipEnv:
    # def __init__(self, size = 8, shaped_reward = False):
    def __init__(self, size, shaped_reward=False, dynamic=False):
        # Initialize env params
        self.shaped_reward = shaped_reward
        self.steps_done = None
        self.max_steps = 200
        self.size = size

        # Initialize the state and target
        self.state = np.array([])
        self.target = np.array([])

        # Dynamic goal settings
        self.dynamic = dynamic
        self.flip_prob = 0.3


    def step(self, action):
        """
        :param action: an int number between 0 and (size - 1) to flip in self.state
        :return: the new state, the reward of making that action (-1: not final, 0: final) and the 'done' bit
        """
        self.state[action] = 1 - self.state[action] # flip the action bit
        self.steps_done += 1

        next_state = np.copy(self.state)

        if all(self.state == self.target):
            # New state is the target
            reward = 0.0
            done = True
        else:
            if self.shaped_reward:
                # Shaped reward: the distance between state and target
                reward = -np.sum(self.state != self.target, dtype=np.float32)
            else:
                reward = -1.0
            # Check if run is done
            if self.steps_done < self.max_steps:
                done = False
            else:
                done = True

        if self.dynamic:
            if random.random() < self.flip_prob:
                random_bit = random.randint(0, self.size - 1)
                self.target[random_bit] = 1 - self.target[random_bit]

        return next_state, reward, done

    def reset(self):
        self.steps_done = 0
        self.state = np.random.randint(2, size=self.size)
        self.target = np.random.randint(2, size=self.size)

        # If target and initial state are equal, regenerate the target.
        while all(self.state == self.target):
            self.target = np.random.randint(2, size=self.size)

        return np.copy(self.state)
