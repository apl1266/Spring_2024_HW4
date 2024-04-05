import numpy as np
import gymnasium as gym
from bettermdptools_edit.algorithms.planner import Planner
from bettermdptools_edit.utils.plots import Plots
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import matplotlib.pyplot as plt
from bettermdptools_edit.algorithms.rl import RL
import time


frozen_lake = gym.make('FrozenLake8x8-v1', render_mode=None)

t=time.time()
V, V_track, pi,pi_track, iter_stop, pi_stop, t_con = Planner(frozen_lake.P).value_iteration(gamma=1, n_iters=2000)
print(time.time()-t)

#plot state values
size=(8,8)
Plots.values_heat_map(V, "Frozen Lake\nValue Iteration State Values", size,"test.png")
Plots.values_heat_map(np.array(list(pi.values())).reshape((8,8)), "Frozen Lake\nValue Iteration State Values", size)

t=time.time()
V_p, V_track_p, pi_p,pi_track_p,iter_stop_p, pi_stop_p, t_con_p = Planner(frozen_lake.P).policy_iteration(gamma=1, n_iters=100,seed=812)
print(time.time()-t)

#plot state values
size=(8,8)
Plots.values_heat_map(V_p, "Frozen Lake\nValue Iteration State Values", size)



plochange=[]
for i in range (V_track.shape[0]):
    plochange.append(np.sum(V_track[i]))

plt.plot(plochange,"r-")
plt.show()

plochange=[]
for i in range (len(pi_track)):
    mat=np.abs(np.array(list(pi_track[i].values())).reshape((8,8))-np.array(list(pi_track[-1].values())).reshape((8,8)))
    mat_edit=mat
    mat_edit[mat > 1] = 1
    plochange.append(np.sum(mat_edit))

plt.plot(plochange,"b-")
plt.show()

plochange=[]
for i in range (V_track_p.shape[0]):
    plochange.append(np.sum(V_track_p[i]))

plt.plot(plochange,"g-")
plt.show()


plochange=[]
for i in range (len(pi_track_p)):
    mat=np.abs(np.array(list(pi_track_p[i].values())).reshape((8,8))-np.array(list(pi_track_p[-1].values())).reshape((8,8)))
    mat_edit=mat
    mat_edit[mat > 1] = 1
    plochange.append(np.sum(mat_edit))

plt.plot(plochange,"b-")
plt.show()
print(pi_stop,pi_stop_p)
if 0:
    Q, V, pi, Q_track, pi_track = RL(frozen_lake).q_learning(epsilon_decay_ratio=0.9,n_episodes=500000)
    Plots.values_heat_map(V, "Frozen Lake\nValue Iteration State Values", size,"test2.png")


if 0:
    plochange=[]
    for i in range (V_track.shape[0]):
        plochange.append(np.sum(V_track_p[i]-V_track[i]))

    plt.plot(plochange,"r-")
    plt.show()

V, V_track, pi,pi_track, iter_stop, pi_stop, t_con = Planner(frozen_lake.P).value_iteration(gamma=1, n_iters=365)