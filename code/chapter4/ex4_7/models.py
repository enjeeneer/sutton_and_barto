import numpy as np
import os
import itertools
from scipy.stats import poisson
from tqdm import tqdm

class Environment():
    def __init__(self, cars=20, x_requests=3, y_requests=4,\
                x_returns=3, y_returns=2, action_space=5, sale=10,\
                move=-2):
        self.cars = cars
        self.x_returns = x_returns
        self.y_returns = y_returns
        self.x_requests = x_requests
        self.y_requests = y_requests
        self.action_space = np.arange(-action_space,action_space+1)
        self.state_space = np.asarray(list(itertools.product(range(self.cars+1),\
                                                            range(self.cars+1))))
        self.sale = sale
        self.move = move

        # instatitate distributions
        self.xret_dist, self.yret_dist, self.xreq_dist, self.yreq_dist = poisson(self.x_returns),\
                                                        poisson(self.y_returns),\
                                                        poisson(self.x_requests),\
                                                        poisson(self.y_requests)

        # calculate discrete probabilites for each rental/request permutation
        self.xret_prob = [abs(self.xret_dist.cdf(i+1) - self.xret_dist.cdf(i)) for i in range(self.cars + 1)]
        self.yret_prob = [abs(self.yret_dist.cdf(i+1) - self.yret_dist.cdf(i)) for i in range(self.cars + 1)]
        self.xreq_prob = [abs(self.xreq_dist.cdf(i+1) - self.xreq_dist.cdf(i)) for i in range(self.cars + 1)]
        self.yreq_prob= [abs(self.yreq_dist.cdf(i+1) - self.yreq_dist.cdf(i)) for i in range(self.cars + 1)]

        self.rental_perms = np.asarray(list(itertools.product(range(self.cars+1),\
                                                            range(self.cars+1))))
        self.request_perms = np.asarray(list(itertools.product(range(self.cars+1),\
                                                            range(self.cars+1))))

    def reset(self):
        self.state = np.random.randint(10, size=(2,))
        return self.state # intialise state for end of day 0

    def expected_reward(self, state, action):
        '''
        given a state and action, what are each of the expected reward
        given all possible next states and their rewards
        '''
        # move the cars and find reward
        action_array = np.array([action,
                                -action], dtype='int64')
        state_ = state + action_array
        self.reward = np.absolute(action) * self.move

        # obtain expected rentals from distributions
        def expected_requests(state, x_dist, y_dist):
            # our expected value for any rental requests < than our state
            x_req = sum(x_dist.pmf(i) * i for i in range(state[0] + 1))
            y_req = sum(y_dist.pmf(i) * i for i in range(state[1] + 1))

            # our expected value for any rental requests > than our state, note we can only fulfill as many as our state
            x_surplus = (1 - x_dist.cdf(state[0])) * state[0]
            y_surplus = (1 - y_dist.cdf(state[1])) * state[1]

            return x_req + x_surplus + y_req + y_surplus

        exp_req = expected_requests(state_, self.xreq_dist, self.yreq_dist)
        self.reward += (exp_req * self.sale)
        return self.reward

    def expected_state_(self, state, action):
        # move the cars
        action_array = np.array([action,
                                -action], dtype='int64')
        state_ = state + action_array

        # expected returned vehicles
        x_ret = sum(self.xret_dist.pmf(i) * i for i in range(self.cars + 1))
        y_ret = sum(self.yret_dist.pmf(i) * i for i in range(self.cars + 1))

        # expected (fullfillable) requested vehicles
        x_req_belowstate = sum(self.xreq_dist.pmf(i) * i for i in range(state[0] + 1))
        y_req_belowstate = sum(self.yreq_dist.pmf(i) * i for i in range(state[1] + 1))
        x_req_abovestate = (1 - self.xreq_dist.cdf(state[0])) * state[0]
        y_req_abovestate = (1 - self.yreq_dist.cdf(state[1])) * state[1]
        x_req = x_req_belowstate + x_req_abovestate
        y_req = y_req_belowstate + y_req_abovestate

        net = np.array([x_ret - x_req,
                        y_ret - x_req])

        state_ = np.round(state_ + net).astype('int')
        state_[state_<0] = 0
        state_[state_>20] = 20

        return state_

        # update state with action

    # def step(self, state, action):
    #
    #     self.reward = np.absolute(action) * self.move # calculate negative reward for movements
    #     self.state_[self.state_>20] = 20 # ensure no more than 20 cars at each location
    #
    #     bool = np.less(self.state_, requests) # check if state is lower than requests
    #     for i, b in enumerate(bool):
    #         if True:
    #             self.reward += self.state_[i] * self.sale
    #             self.state_[i] = 0
    #         else:
    #             self.reward += requests[i] * self.sale
    #             self.state_[i] -= requests[i]
    #
    #     self.state_ = self.state_ + returns
    #
    #     return self.state_, self.reward


class Agent():
    def __init__(self, max_cars, gamma=0.9, theta=0.001):
        self.gamma = 0.9
        self.v = np.zeros((max_cars+1,max_cars+1), dtype='int64')
        self.pi = np.zeros((max_cars+1,max_cars+1), dtype='int64')
        self.theta = theta
        self.stable = False

    def choose_action(self,state):
        self.action = self.pi[state[0], state[1]]
        return self.action

    def evaluate(self, environment):
        delta = 0
        for s in tqdm(reversed(environment.state_space)):
            v = np.copy(self.v[s[0], s[1]]) # old value for v
            a = self.choose_action(s)
            exp_r = environment.expected_reward(state=s, action=a)
            s_ = environment.expected_state_(state=s, action=a)
            self.v[s[0], s[1]] = exp_r + self.gamma*self.v[s_[0], s_[1]]
            delta = max(delta, np.absolute(v - self.v[s[0], s[1]]))
            if delta < self.theta:
                break
        return self.v

    def improve(self, environment):
        self.stable = True
        for s in tqdm(environment.state_space):
            old_a = self.choose_action(s)
            a_vals = np.zeros((11,), dtype='float')

            # get value of each action in action space
            for i, a in enumerate(environment.action_space):
                exp_r = environment.expected_reward(state=s, action=a)
                s_ = environment.expected_state_(state=s, action=a)
                val = exp_r + self.gamma*self.v[s_[0], s_[1]]
                a_vals[i] = val
            max_idx = np.argmax(a_vals)
            a_max = environment.action_space[max_idx]
            self.pi[s[0], s[1]] = a_max
            self.pi[self.pi>5] = 5
            self.pi[self.pi<-5] = -5
            if a_max != old_a:
                self.stable = False

        return self.pi, self.stable
