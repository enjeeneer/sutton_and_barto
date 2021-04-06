from models import Environment, Agent
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
from mpl_toolkits.mplot3d import Axes3D
from itertools import product

if __name__ == '__main__':
    # intialise value function, policy and environment abitrarily
    env = Environment()
    agent = Agent(max_cars=env.cars)
    #
    # v_functions = []
    # policies = []
    #
    #
    # while not agent.stable:
    #     v_functions.append(agent.v)
    #     policies.append(agent.pi)
    #     agent.v = agent.evaluate(environment=env)
    #     agent.pi, agent.stable = agent.improve(environment=env)
    #
    #
    # v_functions.append(agent.v)
    # policies.append(agent.pi)
    #
    # # save data
    # with open('data/value_functions.pickle', 'wb') as f:
    #     pickle.dump(v_functions, f)
    #
    # with open('data/policies.pickle', 'wb') as f:
    #     pickle.dump(policies, f)

    with open('data/value_functions.pickle', 'rb') as f:
        v_functions = pickle.load(f)

    with open('data/policies.pickle', 'rb') as f:
        policies = pickle.load(f)


    # plotting
    x = np.arange(0,env.cars+1,1)
    y = np.arange(0,env.cars+1,1)
    v_star = v_functions[-1].flatten()
    pi_star = policies[-1]

    # matrix
    fig = plt.figure(figsize=(10,7.5))
    ax = fig.add_subplot(121)
    lim = np.max(np.abs(pi_star))
    ax.matshow(pi_star, cmap=plt.cm.bwr, vmin=-lim, vmax=lim)
    ax.set_xticks(x)
    ax.set_yticks(y)
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.set_xlabel("Cars at location x")
    ax.set_ylabel("Cars at location y")
    for x_1, y_1 in env.state_space:
        plt.text(x=x_1, y=y_1, s=pi_star[x_1, y_1], va='center', ha='center', fontsize=8)
    ax.set_title(r'$\pi_*$', fontsize=20)
    plt.savefig('pi_star.png', dpi=300)

    # 3d surface
    ax = fig.add_subplot(122, projection='3d')

    print(x.shape)
    print(y.shape)
    print(v_star.shape)


    surf = ax.plot_trisurf(xs=x, ys=y, zs=v_star, cmap=plt.cm.viridis, linewidth=0.4)
    ax.set_xticks(x)
    ax.set_yticks(y)
    ax.set_zticks(np.arange(0,1000,50))
    ax.set_xlabel('Cars at first location')
    ax.set_ylabel('Cars at second location')
    ax.set_zlabel(r'$V_*$')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig('v_star.png', dpi=300)
