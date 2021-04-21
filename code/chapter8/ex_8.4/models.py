import numpy as np
import itertools
import secrets
from math import sqrt

class Environment():
    def __init__(self, x_length=9, y_length=6):
        self.x_length = x_length
        self.y_length = y_length
        self.timestep = 0
        self.wall_switch = 3000
        self.action_space = np.array([[1,0],[-1,0], [0,1], [0,-1]], dtype='int')
        self.goal = [self.x_length-1, self.y_length-1]

        # define the boundaries and wall
        bottom_perim = np.asarray(list(itertools.product(range(self.x_length), [-1])))
        top_perim = np.asarray(list(itertools.product(range(self.x_length), [self.y_length])))
        left_perim = np.asarray(list(itertools.product([-1], range(self.y_length))))
        right_perim = np.asarray(list(itertools.product([self.x_length], range(self.y_length))))
        perimeter = np.concatenate((bottom_perim, top_perim, left_perim, right_perim), axis=0)
        self.wall1 = np.asarray([[i, 2] for i in range(1,9,1)])
        self.wall2 = np.asarray([[i, 2] for i in range(0,8,1)])

        self.block1 = np.concatenate((perimeter, self.wall1), axis=0)
        self.block2 = np.concatenate((perimeter, self.wall2), axis=0)

    def reset(self):
        return [3,0]

    def step(self, state, a_idx):
        self.timestep += 1
        action = self.action_space[a_idx]
        state_ = state + action
        reward = 0

        def block(s, block):
            for i in block:
                if i[0] == s[0] and i[1] == s[1]:
                    return True
            return False

        blocked = block(state_, self.block1)

        if self.timestep >= self.wall_switch:
            blocked_new = block(state_, self.block2)
            if blocked_new:
                done = False
                return state, reward, done

            elif state_[0] == self.goal[0] and state_[1] == self.goal[1]:
                done = True
                reward = 1
                return state_, reward, done

            else:
                done = False
                return state_, reward, done

        elif blocked:
            done = False
            return state, reward, done

        elif state_[0] == self.goal[0] and state_[1] == self.goal[1]:
            done = True
            reward = 1
            return state_, reward, done

        else:
            done = False
            return state_, reward, done

class Agent():
    def __init__(self, action_space, state_space=[9, 6], epsilon=0.1, alpha=0.1, gamma=0.95, n=50, k=0.001, dyna_q=False, dyna_q_plus=False, dyna_new=False):
        self.action_space = action_space
        self.q = np.zeros((state_space[0], state_space[1], action_space.shape[0]), dtype='float')
        self.ticker = np.zeros((state_space[0], state_space[1], action_space.shape[0]), dtype='int')
        self.model = np.zeros((state_space[0], state_space[1], action_space.shape[0]), dtype=np.ndarray)
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.n = n
        self.k = k
        self.history = []
        self.dyna_q = dyna_q
        self.dyna_q_plus = dyna_q_plus
        self.dyna_new = dyna_new

    def choose_action(self, state):
        if self.dyna_new:
            a_s = self.q[state[0], state[1], ]
            for i, a in enumerate(a_s): # only need the index i
                a_s[i] = a_s[i] + self.k*sqrt(self.ticker[state[0], state[1], i])
            a_idx = np.random.choice(np.where(a_s == a_s.max())[0])
            self.history.append([state, a_idx])
            self.ticker[state[0], state[1], a_idx] = 0
            self.ticker[~state[0], ~state[1], ~a_idx] += 1
            return a_idx

        else:
            if np.random.random_sample() < self.epsilon:
                a_idx = secrets.randbelow(self.action_space.shape[0])
                self.history.append([state, a_idx])
                self.ticker[state[0], state[1], a_idx] = 0
                self.ticker[~state[0], ~state[1], ~a_idx] += 1
                return a_idx
            else:
                a_s = self.q[state[0], state[1], ]
                a_idx = np.random.choice(np.where(a_s == a_s.max())[0]) # argmax with ties broken randomly
                self.history.append([state, a_idx])
                self.ticker[state[0], state[1], a_idx] = 0
                self.ticker[~state[0], ~state[1], ~a_idx] += 1
                return a_idx

    def learn(self, state_, state, a_idx, reward, planning=False): # learning includes updating the value function and the model
        # update q
        a_s_ = self.q[state_[0], state_[1],]
        max_a_ = np.argmax(a_s_)
        max_q_ = self.q[state_[0], state_[1], max_a_]
        q = self.q[state[0], state[1], a_idx]
        self.q[state[0], state[1], a_idx] = q + self.alpha*(reward + self.gamma*max_q_ - q)

        # update model
        if planning==False:
            self.model[state[0], state[1], a_idx] = [reward, state_]

    def plan(self):
        for i in range(self.n):
            sample = secrets.choice(self.history)
            state, a_idx = sample[0], sample[1]
            reward, state_ = self.model[state[0], state[1], a_idx][0], self.model[state[0], state[1], a_idx][1]
            if self.dyna_q_plus:
                reward = reward + self.k*sqrt(self.ticker[state[0], state[1], a_idx])
            self.learn(state_, state, a_idx, reward, planning=True)
