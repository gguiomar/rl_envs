# Reinforcement learning environments

from pylab import *
import matplotlib.pyplot as plt
import numpy as np


class world(object):
    def __init__(self):
        return

    def get_outcome(self, current_state, action=0):

        # Update state
        if current_state < self.T_states - 1:
            next_state = current_state + 1
        else:
            next_state = 0

        # Check for reward
        if self.reward_state == self.state_dict[next_state]:
            reward = self.reward_magnitude
        else:
            reward = 0

        return next_state, reward


class classical_conditioning(world):

    def __init__(self, n_steps):

        # state variables
        self.n_steps = n_steps  # number of time steps in each episode
        self.n_actions = 0
        self.dim_x = n_steps
        self.dim_y = 1

        # reward related variables
        self.reward_state = [0, 0]
        self.reward_probability = reward_probability
        self.reward_time = reward_time
        self.reward_magnitude = reward_magnitude

        # time at which the conditioned stimulus is presented
        self.cs_time = int(n_steps/4) - 1

        # generate the state dictionary
        self.create_state_dictionary()
        # self.create_state_dictionary_2(reward_time)

    def define_reward(self, reward_magnitude, reward_time):
        """
        Define reward state and magnitude of reward
        """

        if reward_time >= self.n_steps - self.cs_time:
            self.reward_magnitude = 0

        else:
            self.reward_magnitude = reward_magnitude
            self.reward_state = [1, reward_time]

    def create_state_dictionary(self):
        """
        This dictionary maps number of iterations in each episode (tsteps)
        to state identities:
        
        step   - 0 1 2 3 4 5       6 7 8 9 10 11 12 ...
        (cs)   - 0 0 0 0 0 0 ( cs) 1 1 1 1  1  1  1 ...
        (t )   - 0 1 2 3 4 5 (...) 0 1 2 3  4  5  6 ...
        """

        c = 0
        d = 0

        self.state_dict = {}
        for s in range(self.n_steps):
            if s < self.cs_time:
                self.state_dict[s] = [0, c]
                c += 1
            else:
                self.state_dict[s] = [1, d]
                d += 1


class n_armed_bandit(world):
    """
    World: N-Armed bandit.
    Only one state, multiple actions.
    Each action returns different amount of reward.
    For each action, rewards are randomly sampled from normal distribution, 
        with a mean associated with that arm and unit variance.
    """

    def __init__(self, arm_number):
        self.name = "n_armed_bandit"
        self.n_states = 1
        self.n_actions = arm_number
        self.dim_x = 1
        self.dim_y = 1

        self.mu = [np.random.normal(0, 1) for a in range(self.n_actions)]

    def get_outcome(self, state, action):

        self.rewards = [np.random.normal(self.mu[i], 1)
                        for i in range(self.n_actions)]
        next_state = None

        reward = self.rewards[action]
        return int(next_state) if next_state is not None else None, reward


class drifting_n_armed_bandit(world):
    """
    World: N-Armed bandit.
    Only one state, multiple actions.
    Each action returns different amounts of rewards.
    For each action, rewards are randomly sampled from normal distribution, 
        with a mean associated with that arm and unit variance.
    In the case of the non-stationary bandit, the mean reward associated with each arm 
        follows a Gaussian random.
    """

    def __init__(self, arm_number, drift):
        self.name = "drifting_n_armed_bandit"
        self.n_states = 1
        self.n_actions = arm_number
        self.dim_x = 1
        self.dim_y = 1

        self.mu_min = 0.25
        self.mu_max = 0.75
        self.drift = drift

        self.mu = [np.random.normal(0, 0) for a in range(self.n_actions)]

    def update_mu(self):
        self.mu += np.random.normal(0, self.drift, self.n_actions)

    def get_outcome(self, state, action):

        self.update_mu()
        self.rewards = [np.random.normal(self.mu[i], 1)
                        for i in range(self.n_actions)]
        next_state = None

        reward = self.rewards[action]
        return int(next_state) if next_state is not None else None, reward


class drifting_probabilitic_bandit(world):
    """
    World: 2-Armed bandit.
    Each arm returns reward with a different probability.
    The probability of returning rewards for all arms follow Gaussian random walks.
    """

    def __init__(self, arm_number, drift):
        self.name = "n_armed_bandit"
        self.n_states = 1
        self.n_actions = arm_number
        self.dim_x = 1
        self.dim_y = 1

        self.mu_min = 0.25
        self.mu_max = 0.75
        self.drift = drift

        self.reward_mag = 1

        self.mu = [np.random.uniform(self.mu_min, self.mu_max)
                   for a in range(self.n_actions)]

    def update_mu(self):
        self.mu += np.random.normal(0, self.drift, self.n_actions)
        self.mu[self.mu > self.mu_max] = self.mu_max
        self.mu[self.mu < self.mu_min] = self.mu_min

    def get_outcome(self, state, action):

        self.update_mu()
        self.rewards = [self.reward_mag if np.random.uniform(
            0, 1) < self.mu[a] else 0 for a in range(self.n_actions)]
        next_state = None

        reward = self.rewards[action]
        return int(next_state) if next_state is not None else None, reward


class contextual_bandits(drifting_probabilitic_bandit):

    def __init__(self):

        self.name = "contextual_bandit"
        self.n_states = 3
        self.n_actions = 2
        self.dim_x = 1
        self.dim_y = 1

        self.n_arms = self.n_actions
        self.n_of_bandits = self.n_states
        self.drift = 0.02
        self.bandits = [n_armed_bandit(self.n_arms, self.drift)
                        for n in range(self.n_of_bandits)]

    def get_outcome(self, state, action):

        _, reward = self.bandits[state].get_outcome(0, action)
        available_states = [s for s in range(self.n_of_bandits) if s != state]
        next_state = np.random.choice(available_states)

        return int(next_state) if next_state is not None else None, reward


class Daw_two_step_task(drifting_probabilitic_bandit):

    def __init__(self):

        self.name = "Daw_two_step_task"
        self.n_states = 3
        self.n_actions = 2
        self.dim_x = 1
        self.dim_y = 1

        self.n_arms = self.n_actions
        self.n_of_bandits = 2
        self.drift = 0.02

        self.context_transition_prob = 0.7
        self.bandits = [drifting_probabilitic_bandit(
            self.n_arms, self.drift) for n in range(self.n_of_bandits)]

    def get_outcome(self, state, action):

        if state == 0:
            reward = 0
            if action == 0:
                if np.random.uniform(0, 1) < self.context_transition_prob:
                    next_state = 1
                else:
                    next_state = 2
            elif action == 1:
                if np.random.uniform(0, 1) < self.context_transition_prob:
                    next_state = 2
                else:
                    next_state = 1
            else:
                print('No valid action specified')

        if state == 1:
            _, reward = self.bandits[0].get_outcome(0, action)
            next_state = 0

        if state == 2:
            _, reward = self.bandits[1].get_outcome(0, action)
            next_state = 0

        return int(next_state) if next_state is not None else None, reward


class cliff_world(world):
    """
    World: Cliff world.
    40 states (4-by-10 grid world).
    The mapping from state to the grids are as follows:
    30 31 32 ... 39
    20 21 22 ... 29
    10 11 12 ... 19
    0  1  2  ...  9
    0 is the starting state (S) and 9 is the goal state (G).
    Actions 0, 1, 2, 3 correspond to right, up, left, down.
    Moving anywhere from state 9 (goal state) will end the session.
    Taking action down at state 11-18 will go back to state 0 and incur a
        reward of -100.
    Landing in any states other than the goal state will incur a reward of -1.
    Going towards the border when already at the border will stay in the same
        place.
    """

    def __init__(self):
        self.name = "cliff_world"
        self.n_states = 40
        self.n_actions = 4
        self.dim_x = 10
        self.dim_y = 4

    def get_outcome(self, state, action):
        if state == 9:  # goal state
            reward = 0
            next_state = None
            return next_state, reward
        reward = -1  # default reward value
        if action == 0:  # move right
            next_state = state + 1
            if state % 10 == 9:  # right border
                next_state = state
            elif state == 0:  # start state (next state is cliff)
                next_state = None
                reward = -100
        elif action == 1:  # move up
            next_state = state + 10
            if state >= 30:  # top border
                next_state = state
        elif action == 2:  # move left
            next_state = state - 1
            if state % 10 == 0:  # left border
                next_state = state
        elif action == 3:  # move down
            next_state = state - 10
            if state >= 11 and state <= 18:  # next is cliff
                next_state = None
                reward = -100
            elif state <= 9:  # bottom border
                next_state = state
        else:
            print("Action must be between 0 and 3.")
            next_state = None
            reward = None
        return int(next_state) if next_state is not None else None, reward


class quentins_world(world):
    """
    World: Quentin's world.
    100 states (10-by-10 grid world).
    The mapping from state to the grid is as follows:
    90 ...       99
    ...
    40 ...       49
    30 ...       39
    20 21 22 ... 29
    10 11 12 ... 19
    0  1  2  ...  9
    54 is the start state.
    Actions 0, 1, 2, 3 correspond to right, up, left, down.
    Moving anywhere from state 99 (goal state) will end the session.
    Landing in red states incurs a reward of -1.
    Landing in the goal state (99) gets a reward of 1.
    Going towards the border when already at the border will stay in the same
        place.
    """

    def __init__(self):
        self.name = "quentins_world"
        self.n_states = 100
        self.n_actions = 4
        self.dim_x = 10
        self.dim_y = 10

    def get_outcome(self, state, action):
        if state == 99:  # goal state
            reward = 0
            next_state = None
            return next_state, reward
        reward = 0  # default reward value
        if action == 0:  # move right
            next_state = state + 1
            if state == 98:  # next state is goal state
                reward = 1
            elif state % 10 == 9:  # right border
                next_state = state
            elif state in [11, 21, 31, 41, 51, 61, 71,
                           12, 72,
                           73,
                           14, 74,
                           15, 25, 35, 45, 55, 65, 75]:  # next state is red
                reward = -1
        elif action == 1:  # move up
            next_state = state + 10
            if state == 89:  # next state is goal state
                reward = 1
            if state >= 90:  # top border
                next_state = state
            elif state in [2, 12, 22, 32, 42, 52, 62,
                           3, 63,
                           64,
                           5, 65,
                           6, 16, 26, 36, 46, 56, 66]:  # next state is red
                reward = -1
        elif action == 2:  # move left
            next_state = state - 1
            if state % 10 == 0:  # left border
                next_state = state
            elif state in [17, 27, 37, 47, 57, 67, 77,
                           16, 76,
                           75,
                           14, 74,
                           13, 23, 33, 43, 53, 63, 73]:  # next state is red
                reward = -1
        elif action == 3:  # move down
            next_state = state - 10
            if state <= 9:  # bottom border
                next_state = state
            elif state in [22, 32, 42, 52, 62, 72, 82,
                           23, 83,
                           84,
                           25, 85,
                           26, 36, 46, 56, 66, 76, 86]:  # next state is red
                reward = -1
        else:
            print("Action must be between 0 and 3.")
            next_state = None
            reward = None
        return int(next_state) if next_state is not None else None, reward


class windy_cliff_grid(world):
    """
    World: Windy grid world with cliffs.
    84 states(6-by-14 grid world).
    4 possible actions.
    Actions 0, 1, 3, 4 correspond to right, up, left, down.
    Each action returns different amount of reward.
    """

    def __init__(self):
        self.name = "windy_cliff_grid"
        self.n_states = 168
        self.n_actions = 4
        self.dim_x = 14
        self.dim_y = 12

    def get_outcome(self, state, action):
        """
        Obtains the outcome (next state and reward) obtained when taking
            a particular action from a particular state.
        Args:
            state: int, current state.
            action: int, action taken.
        Returns:
            the next state (int) and the reward (int) obtained.
        """
        next_state = None
        reward = 0
        if state in [53, 131]:  # end of MDP
            return next_state, reward
        if action == 0:  # move right
            next_state = state + 1
            if state == 38:  # goal state 1
                next_state = 53
                reward = 100
            elif state == 158:  # goal state 2
                next_state = 131
                reward = 100
            elif state == 1:  # cliff
                next_state = None
                reward = -100
            # room 1 wind
            elif 7 <= state <= 51 and (state % 14 == 7 or state % 14 == 8 or state % 14 == 9):
                next_state = state + 29
            elif state in [63, 64, 65]:  # room 1 wind
                next_state = state + 15
            # room 1 wind
            elif 10 <= state <= 68 and (state % 14 == 10 or state % 14 == 11 or state % 14 == 12):
                next_state = state + 15
            # room 2 wind
            elif 113 <= state <= 157 and (state % 14 == 1 or state % 14 == 2 or state % 14 == 3):
                next_state = state - 13
            # room 2 wind
            elif 130 <= state <= 160 and (state % 14 == 4 or state % 14 == 5 or state % 14 == 6):
                next_state = state - 27
            elif state in [116, 117, 118]:  # room 2 wind
                next_state = state - 13
            elif 19 <= state <= 75 and state % 14 == 5:  # room 1 left border
                next_state = state
            elif 105 <= state <= 161 and state % 14 == 7:  # room 2 right border
                next_state = state
            elif state % 14 == 13:  # world right border
                next_state = state
        elif action == 1:  # move up
            next_state = state - 14
            if state in [16, 17, 18, 84]:  # cliff
                next_state = None
                reward = -100
            # room 1 wind
            elif 21 <= state <= 65 and (state % 14 == 7 or state % 14 == 8 or state % 14 == 9):
                next_state = state + 14
            elif state in [7, 8, 9]:  # room 1 wind
                next_state = state + 28
            elif state in [77, 78, 79]:  # room 1 wind
                next_state = state
            # room 1 wind
            elif 24 <= state <= 82 and (state % 14 == 10 or state % 14 == 11 or state % 14 == 12):
                next_state = state
            elif state in [10, 11, 12]:  # room 1 wind
                next_state = state + 14
            # room 2 wind
            elif 127 <= state <= 157 and (state % 14 == 1 or state % 14 == 2 or state % 14 == 3):
                next_state = state - 28
            # room 2 wind
            elif 144 <= state <= 160 and (state % 14 == 4 or state % 14 == 5 or state % 14 == 6):
                next_state = state - 42
            elif state in [130, 131, 132]:  # room 2 wind
                next_state = state - 28
            elif 90 <= state <= 97:  # room 1 bottom border
                next_state = state
            elif 99 <= state <= 105:  # room 2 top border
                next_state = state
            elif 0 <= state <= 13:  # world top border
                next_state = state
        elif action == 2:  # move left
            next_state = state - 1
            if state == 40:  # goal state 1
                next_state = 53
                reward = 100
            elif state == 160:  # goal state 2
                next_state = 131
                reward = 100
            elif state in [29, 43, 57, 71, 5]:  # cliff
                next_state = None
                reward = -100
            # room 1 wind
            elif 7 <= state <= 51 and (state % 14 == 7 or state % 14 == 8 or state % 14 == 9):
                next_state = state + 27
            elif state in [63, 64, 65]:  # room 1 wind
                next_state = state + 13
            # room 1 wind
            elif 10 <= state <= 68 and (state % 14 == 10 or state % 14 == 11 or state % 14 == 12):
                next_state = state + 13
            # room 2 wind
            elif 113 <= state <= 157 and (state % 14 == 1 or state % 14 == 2 or state % 14 == 3):
                next_state = state - 15
            elif state == 99:  # room 2 wind
                next_state = state - 15
            # room 2 wind
            elif 130 <= state <= 160 and (state % 14 == 4 or state % 14 == 5 or state % 14 == 6):
                next_state = state - 29
            elif state in [116, 117, 118]:  # room 2 wind
                next_state = state - 15
            elif 20 <= state <= 76 and state % 14 == 6:  # room 1 left border
                next_state = state
            elif 106 <= state <= 162 and state % 14 == 8:  # room 2 right border
                next_state = state
            elif state % 14 == 0:  # world left border
                next_state = state
        elif action == 3:  # move down
            next_state = state + 14
            if state == 25:  # goal state 1
                next_state = 53
                reward = 100
            elif state == 145:  # goal state 2
                next_state = 131
                reward = 100
            elif state == 14:  # cliff
                next_state = None
                reward = -100
            # room 1 wind
            elif 7 <= state <= 37 and (state % 14 == 7 or state % 14 == 8 or state % 14 == 9):
                next_state = state + 42
            elif state in [49, 50, 51]:  # room 1 wind
                next_state = state + 28
            # room 2 wind
            elif 99 <= state <= 143 and (state % 14 == 1 or state % 14 == 2 or state % 14 == 3):
                next_state = state
            elif state in [155, 156, 157]:  # room 2 wind
                next_state = state - 14
            # room 2 wind
            elif 116 <= state <= 146 and (state % 14 == 4 or state % 14 == 5 or state % 14 == 6):
                next_state = state - 14
            elif state in [102, 103, 104]:  # room 2 wind
                next_state = state
            elif state in [158, 159, 160]:  # room 2 wind
                next_state = state - 28
            elif 76 <= state <= 83:  # room 1 bottom border
                next_state = state
            elif 85 <= state <= 91:  # room 2 top border
                next_state = state
            elif 154 <= state <= 167:  # world bottom border
                next_state = state
        else:
            print("Action must be between 0 and 3.")
            next_state = None
            reward = None
        return int(next_state) if next_state is not None else None, reward


class windy_cliff_grid_2(windy_cliff_grid):
    def get_outcome(self, state, action):
        """
        Obtains the outcome (next state and reward) obtained when taking
            a particular action from a particular state.
        Args:
            state: int, current state.
            action: int, action taken.
        Returns:
            the next state (int) and the reward (int) obtained.
        """
        next_state = None
        reward = 0
        if state in [53, 131]:  # end of MDP
            return next_state, reward
        if action == 0:  # move right
            next_state = state + 1
            if state == 38:  # goal state 1
                next_state = 53
                reward = 100
            elif state == 158:  # goal state 2
                next_state = 131
                reward = 100
            elif state == 1:  # cliff
                next_state = None
                reward = -100
            # room 1 wind
            elif 7 <= state <= 51 and (state % 14 == 7 or state % 14 == 8 or state % 14 == 9):
                next_state = state + 29
            elif state in [63, 64, 65]:  # room 1 wind
                next_state = state + 15
            # room 1 wind
            elif 10 <= state <= 68 and (state % 14 == 10 or state % 14 == 11 or state % 14 == 12):
                next_state = state + 15
            # room 2 wind
            elif 113 <= state <= 157 and (state % 14 == 1 or state % 14 == 2 or state % 14 == 3):
                next_state = state - 13
            # room 2 wind
            elif 130 <= state <= 160 and (state % 14 == 4 or state % 14 == 5 or state % 14 == 6):
                next_state = state - 27
            elif state in [116, 117, 118]:  # room 2 wind
                next_state = state - 13
            elif 5 <= state <= 75 and state % 14 == 5:  # room 1 left border
                next_state = state
            elif 105 <= state <= 147 and state % 14 == 7:  # room 2 right border
                next_state = state
            elif state % 14 == 13:  # world right border
                next_state = state
        elif action == 1:  # move up
            next_state = state - 14
            if state in [16, 17, 18, 84]:  # cliff
                next_state = None
                reward = -100
            # room 1 wind
            elif 21 <= state <= 65 and (state % 14 == 7 or state % 14 == 8 or state % 14 == 9):
                next_state = state + 14
            elif state in [7, 8, 9]:  # room 1 wind
                next_state = state + 28
            elif state in [77, 78, 79]:  # room 1 wind
                next_state = state
            # room 1 wind
            elif 24 <= state <= 82 and (state % 14 == 10 or state % 14 == 11 or state % 14 == 12):
                next_state = state
            elif state in [10, 11, 12]:  # room 1 wind
                next_state = state + 14
            # room 2 wind
            elif 127 <= state <= 157 and (state % 14 == 1 or state % 14 == 2 or state % 14 == 3):
                next_state = state - 28
            # room 2 wind
            elif 144 <= state <= 160 and (state % 14 == 4 or state % 14 == 5 or state % 14 == 6):
                next_state = state - 42
            elif state in [130, 131, 132]:  # room 2 wind
                next_state = state - 28
            elif 90 <= state <= 96:  # room 1 bottom border
                next_state = state
            elif 98 <= state <= 105:  # room 2 top border
                next_state = state
            elif 0 <= state <= 13:  # world top border
                next_state = state
        elif action == 2:  # move left
            next_state = state - 1
            if state == 40:  # goal state 1
                next_state = 53
                reward = 100
            elif state == 160:  # goal state 2
                next_state = 131
                reward = 100
            elif state in [29, 43, 57, 71, 5]:  # cliff
                next_state = None
                reward = -100
            # room 1 wind
            elif 7 <= state <= 51 and (state % 14 == 7 or state % 14 == 8 or state % 14 == 9):
                next_state = state + 27
            elif state in [63, 64, 65]:  # room 1 wind
                next_state = state + 13
            # room 1 wind
            elif 10 <= state <= 68 and (state % 14 == 10 or state % 14 == 11 or state % 14 == 12):
                next_state = state + 13
            # room 2 wind
            elif 113 <= state <= 157 and (state % 14 == 1 or state % 14 == 2 or state % 14 == 3):
                next_state = state - 15
            elif state == 99:  # room 2 wind
                next_state = state - 15
            # room 2 wind
            elif 130 <= state <= 160 and (state % 14 == 4 or state % 14 == 5 or state % 14 == 6):
                next_state = state - 29
            elif state in [116, 117, 118]:  # room 2 wind
                next_state = state - 15
            elif 6 <= state <= 76 and state % 14 == 6:  # room 1 left border
                next_state = state
            elif 106 <= state <= 148 and state % 14 == 8:  # room 2 right border
                next_state = state
            elif state % 14 == 0:  # world left border
                next_state = state
        elif action == 3:  # move down
            next_state = state + 14
            if state == 25:  # goal state 1
                next_state = 53
                reward = 100
            elif state == 145:  # goal state 2
                next_state = 131
                reward = 100
            elif state == 14:  # cliff
                next_state = None
                reward = -100
            # room 1 wind
            elif 7 <= state <= 37 and (state % 14 == 7 or state % 14 == 8 or state % 14 == 9):
                next_state = state + 42
            elif state in [49, 50, 51]:  # room 1 wind
                next_state = state + 28
            # room 2 wind
            elif 99 <= state <= 143 and (state % 14 == 1 or state % 14 == 2 or state % 14 == 3):
                next_state = state
            elif state in [155, 156, 157]:  # room 2 wind
                next_state = state - 14
            # room 2 wind
            elif 116 <= state <= 146 and (state % 14 == 4 or state % 14 == 5 or state % 14 == 6):
                next_state = state - 14
            elif state in [102, 103, 104]:  # room 2 wind
                next_state = state
            elif state in [158, 159, 160]:  # room 2 wind
                next_state = state - 28
            elif 76 <= state <= 82:  # room 1 bottom border
                next_state = state
            elif 84 <= state <= 91:  # room 2 top border
                next_state = state
            elif 154 <= state <= 167:  # world bottom border
                next_state = state
        else:
            print("Action must be between 0 and 3.")
            next_state = None
            reward = None
        return int(next_state) if next_state is not None else None, reward
