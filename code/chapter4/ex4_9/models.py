import numpy as np
import os

class environment():
    def steps(self, state, action):
        self.state_win = state + action
        self.state_lose = state - action
        return self.state_win, self.state_lose

class agent():
    def __init__(self):
        self.states = 101
        self.v = np.zeros((self.states), dtype='float')
        self.v[-1] = 1
        self.stable = False
        self.theta = 0.001

    def possible_actions(self, state):
        actions = np.arange(1, min(state,100-state)+1, 1)
        return actions

    def value_iteration(self, ph, env):
        delta = self.theta
        sweeps = []
        sweep = 0
        while delta >= self.theta:
            old_values = self.v.copy()
            for state in range(1,self.states-1):
                values = []
                actions = self.possible_actions(state)
                for a in actions:
                    state_win, state_lose = env.steps(state, a)
                    value = (ph * self.v[state_win]) + ((1 - ph) * self.v[state_lose])
                    values.append(value)

                values = np.array(values)
                self.v[state] = np.amax(values)  # update value function with value maximising action
            sweeps.append(old_values)
            sweep += 1
            delta = np.max(np.abs(old_values - self.v))

            print(f"Probability of Heads: {ph}")
            print(f"End of sweep: {sweep}, Delta = {delta}")

        return self.v, sweeps

    def find_policy(self, v,  env, ph):
        stakes = []
        for state in range(1,self.states-1):
            a_vals = []
            actions = self.possible_actions(state)
            for a in actions:
                state_win, state_lose = env.steps(state, a)
                a_val = (ph * v[state_win]) + ((1 - ph) * v[state_lose])
                a_vals.append(a_val)

            a_arr = np.array(a_vals)
            a_max = np.argmax(a_arr) + 1
            stakes.append(a_max)

        return stakes
