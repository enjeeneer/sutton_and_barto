import numpy as np
import random

class Env():
    def __init__(self, states=10000, terminate=0.1, b=1):
        self.random_state = np.random.RandomState(seed=0)
        self.states = np.arange(states)
        states_1 = random.sample(range(states), len(range(states)))
        self.states_1 = np.array([states_1, reversed(states_1)])
        self.states_3 = []
        self.terminate = terminate
        self.b = b
        self.rewards = self.random_state.normal(size=(states, 2))
        self.rewards[0][0], self.rewards[0][1] = 0, 0

        if self.b==3:
            for _ in self.states:
                sample = np.random.choice(self.states, 3, replace=False)
                self.states_3.append(sample)

            self.states_3 = np.array([self.states_3, reversed(self.states_3)])

    def step(self, state, action, on_policy=False, uniform=False):
        if on_policy:
            if np.random.random_sample() < self.terminate:
                state_ = 0
                reward = 0
                done = True
                return state_, reward, done

            elif self.b==3:
                state_ = np.random.choice(self.states_3[action,state])
                reward = np.random.choice(self.rewards[state_])
                done = False
                return state_, reward, done

            else:
                state_ = self.states_1[state]
                reward = np.random.choice(self.rewards[state_])
                done = False
                return state_, reward, done

        else:
            if self.b==3:
                states_ = self.states_3[state]
                reward = []
                for s_ in states_:
                    r = self.rewards[s_]
                    reward.append(r)
                return states_, reward

            else:
                state_ = self.states_1[state]
                reward = np.random.choice(self.rewards[state_])
                return state_, reward

    def reset(self):
        state = 0
        return state


class Agent():
    def __init__(self, states=10000, epsilon=0.1, alpha=0.1, theta=0.001, uniform=False, on_policy=False):
        self.epsilon = epsilon
        self.alpha = alpha
        self.theta = theta
        self.actions = np.array([0,1], dtype='int')
        self.q = np.zeros((states, 2), dtype='float')
        self.uniform = uniform
        self.on_policy = on_policy
        self.expected_updates = 0
        self.v_on_policy = 0
        self.v_uniform = 0

    def choose_action(self, state, greedy=False):
        if greedy:
            a_s = self.q[state,]
            a_idx = np.random.choice(np.where(a_s == a_s.max())[0]) # argmax with ties broken randomly
            return a_idx

        else:
            if np.random.random_sample() < self.epsilon:
                a_idx = np.random.choice(self.actions)
                return a_idx

            else:
                a_s = self.q[state,]
                a_idx = np.random.choice(np.where(a_s == a_s.max())[0]) # argmax with ties broken randomly
                return a_idx

    def learn(self, state_, state, action, reward, b, on_policy=False, uniform=False):
        if on_policy:
            q = self.q[state, action]
            a_s_ = self.q[state_, ]
            max_a_ = np.argmax(a_s_)
            q_ = self.q[state_, max_a_]
            self.q[state, action] = q + self.alpha*(reward + q_ - q)
            self.expected_updates += 1

        else:
            p = 1/b
            a = np.random.choice(self.actions) # action is independent of outcome
            e_update = []
            if b == 1:
                a_s_ = self.q[state_, ]
                max_a_ = np.argmax(a_s_)
                q_ = self.q[state_, max_a_]
                e = p*(reward + q_)
                e_update.append(e)
                self.expected_updates += 1

            else:
                for i, s_ in enumerate(state_):
                    r = reward[i]
                    a_s_ = self.q[s_, ]
                    max_a_ = np.argmax(a_s_)
                    q_ = self.q[s_, max_a_]
                    e = p*(r + q_)
                    e_update.append(e)
                    self.expected_updates += 1
                self.q[state, a] = np.sum(e_update)

    def evaluate(self, env):
        delta = 0
        s = env.states[0]
        for _ in range(10000): # some abritraliy large number
            a = self.choose_action(s, greedy=True)
            v = np.copy(self.q[s, a]) # old value for v under greedy policy
            s_, reward, done = env.step(s, on_policy=True)
            a_ = self.choose_action(s_, greedy=True)
            v_ = self.q[s_, a_]
            self.v = reward + v_
            delta = max(delta, np.absolute(v - self.v))
            if delta < self.theta:
                break
        return self.v
