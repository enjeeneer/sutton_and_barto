from models import Environment, Agent
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

if __name__ == '__main__':
    # intialise value function, policy and environment abitrarily
    env = Environment()
    agent = Agent(max_cars=env.cars)

    v_functions = []
    policies = []


    while not agent.stable:
        v_functions.append(agent.v)
        policies.append(agent.pi)
        agent.v = agent.evaluate(environment=env)
        agent.pi, agent.stable = agent.improve(environment=env)


    v_functions.append(agent.v)
    policies.append(agent.pi)

    # plot surface
    x = np.arange(1,env.cars+1,1)
    y = np.arange(1,env.cars+1,1)
    z_star = v_functions[-1]
    pi_star = policies[-1]

    fig = plt.figure(figsize=(10,7.5))
    lim = np.max(np.abs(pi_star))
    ax.matshow(pi_star, cmap=plt.cm.bwr, vmin=-lim, vmax=lim)
    ax.set_xticks(range(env.cars))
    ax.set_yticks(range(env.cars))
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.set_xlabel("Cars at location x")
    ax.set_ylabel("Cars at location y")
    ax.set_xticks([x - 0.5 for x in range(1, env.cars)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, env.cars)], minor=True)
    for x, y in product(range(env.cars), range(env.cars)):
        ax.text(x=x, y=y, s=int(pi_star[x, y]), va='center', ha='center', fontsize=8)
    ax.set_title(r'$\pi_*$', fontsize=20)
    plt.savefig(pi_star.png, dpi=300)


    # ax = fig.gca(projection='3d')
    # surf = ax.plot_trisurf(xs=x, ys=y, zs=z, cmap=plt.cm.viridis, linewidth=0.4)
    # ax.set_xticks(np.arange(1,21,20))
    # ax.set_yticks(np.arange(1,21,20))
    # ax.set_zticks(np.arange(0,1000,50))
    # ax.set_xlabel('Cars at first location')
    # ax.set_ylabel('Cars at second location')
    # ax.set_zlabel('V_pi')
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.savefig('v_star.png', dpi=300)
