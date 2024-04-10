import numpy as np
import gymnasium as gym
from bettermdptools_edit.algorithms.planner import Planner
from bettermdptools_edit.utils.plots import Plots
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import matplotlib.pyplot as plt
from bettermdptools_edit.algorithms.rl import RL
from bettermdptools.utils.blackjack_wrapper import BlackjackWrapper
from bettermdptools.utils.test_env import TestEnv
import time

def plot(y,name,y_label,x_label="number of iteration"):
    plt.plot(y, label=name)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(name)
    plt.savefig(name+".png")
    plt.clf()

def plot2(x,y,name,y_label,x_label="gamma"):
    plt.plot(x,y,"-o", label=name)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(name)
    plt.savefig(name+".png")
    plt.clf()

def plot3(y,yy,name,y_label,x_label="number of iteration"):
    plt.plot(y, label="Convergence to final policy")
    plt.plot(yy, label="Convergence to true policy")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(name)
    plt.legend()
    plt.savefig(name+".png")
    plt.clf()

def plot4(x,y,yy,name,y_label,x_label="gamma"):
    plt.plot(x,y,"-o", label="State Values convergence")
    plt.plot(x,yy,"-o", label="Policy convergence")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(name)
    plt.legend()
    plt.savefig(name+".png")
    plt.clf()

def plot5(y,name,y_label,x_label="number of iteration"):
    plt.plot(y[0], label="gamma=1, epsilon=1, alpha=0.1")
    plt.plot(y[1], label="gamma=1, epsilon=0.5, alpha=0.1")
    plt.plot(y[2], label="gamma=1, epsilon=0.45, alpha=0.1")
    plt.plot(y[3], label="gamma=1, epsilon=0.5, alpha=0.3")
    plt.plot(y[4], label="gamma=1, epsilon=0.5, alpha=0.03")
    plt.plot(y[5], label="gamma=0.9, epsilon=0.5, alpha=0.1")
    plt.plot(y[6], label="gamma=0.3, epsilon=0.5, alpha=0.1")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(name)
    plt.legend()
    plt.savefig(name+".png")
    plt.clf()



frozen_lake = gym.make('FrozenLake8x8-v1', render_mode=None)
size=(8,8)


if 1:
    for alg in ("VI","PI"):
        print("--------------------")
        print(alg+" data bellow")
        print("--------------------")
        xx=[1,0.99,0.9,0.3]
        v_t=[]
        v_i=[]
        p_t=[]
        p_i=[]
        for g in [(1,"1"),(0.99,"099"),(0.9,"09"),(0.3,"03")]:
            print("--------------------")
            if alg=="VI":
                V, V_track, pi, pi_track, iter_stop, pi_stop, t_con = Planner(frozen_lake.P).value_iteration(gamma=g[0], n_iters=2000)
            else:
                V, V_track, pi, pi_track, iter_stop, pi_stop, t_con = Planner(frozen_lake.P).policy_iteration(gamma=g[0],n_iters=40,seed=812)
                if g[0]==0.3:
                    print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
                    print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
                    print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
                    print(V[0],"value of start state at PI and gamma=0.3")
                    print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
                    print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
                    print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
            v_t.append(t_con)
            v_i.append(iter_stop)
            p_i.append(pi_stop)
            print(t_con, "seconds to converge "+alg+" frozen lake at gamma="+g[1]+", and standard epsilon")
            print(iter_stop, "iteration to converge states values to standard epsilon")
            print(pi_stop, "iteration to converge to stable policy")
            Plots.values_heat_map(V, "Frozen Lake "+alg+"\nState Values gamma="+g[1]+"", size, "frozen_lake_"+alg+"_V_Gamma_"+g[1]+".png")
            Plots.values_heat_map(np.array(list(pi.values())).reshape((8, 8)), "Frozen Lake "+alg+"\npolicy gamma="+g[1], size,
                                  "frozen_lake_"+alg+"_P_Gamma_"+g[1]+".png")

            plochange = []
            for i in range(V_track.shape[0]):
                plochange.append(np.sum(V_track[i]) / 64)
            plot(plochange, "frozen_lake_"+alg+"_mean_state_values_gamma_"+g[1], "mean state value")

            plochange = []
            for i in range(len(pi_track)):
                mat = np.abs(np.array(list(pi_track[i].values())).reshape((8, 8)) - np.array(list(pi_track[-1].values())).reshape((8, 8)))
                mat_edit = mat
                mat_edit[mat > 1] = 1
                plochange.append(np.sum(mat_edit))
            plot(plochange, "frozen_lake_"+alg+"_policy_gamma_"+g[1], "number of varied states from converged policy")

            if alg == "Vi":
                V, V_track, pi, pi_track, iter_stop, pi_stop, t_con = Planner(frozen_lake.P).value_iteration(gamma=g[0], n_iters=pi_stop)
            else:
                V, V_track, pi, pi_track, iter_stop, pi_stop, t_con = Planner(frozen_lake.P).policy_iteration(gamma=g[0],n_iters=pi_stop,seed=812)
            p_t.append(t_con)
            print(t_con, "seconds to converge "+alg+" frozen lake at gamma="+g[1]+", and standard epsilon until stable policy")
        plot2(xx, v_t, "frozen_lake_" + alg + "_value_convergence_time", "seconds to converged converged values","gamma")
        plot2(xx, p_t, "frozen_lake_" + alg + "_policy_convergence_time", "seconds to converged converged policy","gamma")
        plot2(xx, v_i, "frozen_lake_" + alg + "_value_convergence_iterations", "iterations to converged converged values","gamma")
        plot2(xx, p_i, "frozen_lake_" + alg + "_policy_convergence_iterations", "iterations to converged converged policy","gamma")

V_V=[]
P_P=[]
P_P_t=[]

if 1:
    iters=300000

    V_VI, V_track_VI, pi_VI, pi_track_VI, iter_stop_VI, pi_stop_VI, t_con_VI = Planner(frozen_lake.P).value_iteration(gamma=1, n_iters=500)
    for e in ((1,"1"), (0.5,"05"), (0.45,"045")):
        Q, V, pi, Q_track, pi_track, V_track = RL(frozen_lake).q_learning(gamma=1,init_alpha=0.1,min_alpha=0.1,init_epsilon=e[0],min_epsilon=e[0], n_episodes=iters,seed=812)
        mat = np.abs(pi_track[-1] - np.array(list(pi_track_VI[-1].values())))
        mat_edit = mat
        mat_edit[mat > 1] = 1
        print(np.sum(mat_edit),"policy difference in between Q Learning and VI, gamma=1,epsilon="+e[1]+" alpha=0.1")
        Plots.values_heat_map(V, "Frozen Lake Q_Learning\nState Values gamma=1,epsilon="+e[1]+"_alpha=0.1", size,"frozen_lake_Q_Learning_V_Gamma_1_epsilon_"+e[1]+"_alpha_01.png")
        Plots.values_heat_map(np.array(list(pi.values())).reshape((8, 8)), "Frozen Lake Q_Learning\npolicy gamma=1,epsilon="+e[1]+"_alpha=0.1",size,"frozen_lake_Q_Learning_P_Gamma_1_epsilon_"+e[1]+"_alpha_01.png")
        plochange = []
        for i in range(len(V_track)):
            plochange.append(V_track[i]/ 64)
        plot(plochange, "frozen_lake_Q_Learning_mean_state_values_gamma_1_epsilon_"+e[1]+"_alpha_01", "mean state value")

        V_V.append(plochange)

        plochange = []
        for i in range(len(pi_track)):
            mat = np.abs(pi_track[i]-pi_track[-1])
            mat_edit = mat
            mat_edit[mat > 1] = 1
            plochange.append(np.sum(mat_edit))
        #plot(plochange, "frozen_lake_Q_Learning_policy_gamma_1", "number of varied states from converged policy")


        plochange_t = []
        for i in range(len(pi_track)):
            mat = np.abs(pi_track[i] - np.array(list(pi_track_VI[-1].values())))
            mat_edit = mat
            mat_edit[mat > 1] = 1
            plochange_t.append(np.sum(mat_edit))
        #plot(plochange_t, "frozen_lake_Q_Learning_true_policy_gamma_1", "number of varied states from converged to true policy")
        plot3(plochange,plochange_t, "frozen_lake_Q_Learning_policy_gamma_1_epsilon_"+e[1]+"_alpha_01", "number of varied states from converged(true) policy")

        P_P.append(plochange)
        P_P_t.append(plochange_t)

if 1:
    iters=300000

    V_VI, V_track_VI, pi_VI, pi_track_VI, iter_stop_VI, pi_stop_VI, t_con_VI = Planner(frozen_lake.P).value_iteration(gamma=1, n_iters=500)
    for a in ((0.3,"03"), (0.03,"003")):
        Q, V, pi, Q_track, pi_track, V_track = RL(frozen_lake).q_learning(gamma=1,init_alpha=a[0],min_alpha=a[0],init_epsilon=0.5,min_epsilon=0.5, n_episodes=iters,seed=812)
        mat = np.abs(pi_track[-1] - np.array(list(pi_track_VI[-1].values())))
        mat_edit = mat
        mat_edit[mat > 1] = 1
        print(np.sum(mat_edit),"policy difference in between Q Learning and VI, gamma=1,epsilon=0.5 alpha="+a[1])
        Plots.values_heat_map(V, "Frozen Lake Q_Learning\nState Values gamma=1,epsilon=0.5_alpha="+a[1], size,"frozen_lake_Q_Learning_V_Gamma_1_epsilon_05_alpha_"+a[1]+".png")
        Plots.values_heat_map(np.array(list(pi.values())).reshape((8, 8)), "Frozen Lake Q_Learning\npolicy gamma=1,epsilon=0.5_alpha="+a[1],size,"frozen_lake_Q_Learning_P_Gamma_1_epsilon_05_alpha_"+a[1]+".png")
        plochange = []
        for i in range(len(V_track)):
            plochange.append(V_track[i]/ 64)
        plot(plochange, "frozen_lake_Q_Learning_mean_state_values_gamma_1_epsilon_05_alpha_"+a[1], "mean state value")

        V_V.append(plochange)

        plochange = []
        for i in range(len(pi_track)):
            mat = np.abs(pi_track[i]-pi_track[-1])
            mat_edit = mat
            mat_edit[mat > 1] = 1
            plochange.append(np.sum(mat_edit))
        #plot(plochange, "frozen_lake_Q_Learning_policy_gamma_1", "number of varied states from converged policy")

        plochange_t = []
        for i in range(len(pi_track)):
            mat = np.abs(pi_track[i] - np.array(list(pi_track_VI[-1].values())))
            mat_edit = mat
            mat_edit[mat > 1] = 1
            plochange_t.append(np.sum(mat_edit))
        #plot(plochange_t, "frozen_lake_Q_Learning_true_policy_gamma_1", "number of varied states from converged to true policy")
        plot3(plochange,plochange_t, "frozen_lake_Q_Learning_policy_gamma_1_epsilon_05_alpha_"+a[1], "number of varied states from converged(true) policy")

        P_P.append(plochange)
        P_P_t.append(plochange_t)


if 1:
    iters=300000

    for g in ((0.9,"09"), (0.3,"03")):
        V_VI, V_track_VI, pi_VI, pi_track_VI, iter_stop_VI, pi_stop_VI, t_con_VI = Planner(frozen_lake.P).value_iteration(gamma=g[0], n_iters=500)
        Q, V, pi, Q_track, pi_track, V_track = RL(frozen_lake).q_learning(gamma=g[0],init_alpha=0.1,min_alpha=0.1,init_epsilon=0.5,min_epsilon=0.5, n_episodes=iters,seed=812)
        mat = np.abs(pi_track[-1] - np.array(list(pi_track_VI[-1].values())))
        mat_edit = mat
        mat_edit[mat > 1] = 1
        print(np.sum(mat_edit),"policy difference in between Q Learning and VI, gamma="+g[1]+",epsilon=0.5 alpha=0.1")
        Plots.values_heat_map(V, "Frozen Lake Q_Learning\nState Values gamma="+g[1]+",epsilon=0.5 alpha=0.1", size,"frozen_lake_Q_Learning_V_Gamma_"+g[1]+"_epsilon_05_alpha_0.1.png")
        Plots.values_heat_map(np.array(list(pi.values())).reshape((8, 8)), "Frozen Lake Q_Learning\npolicy gamma="+g[1]+",epsilon=0.5 alpha=0.1",size,"frozen_lake_Q_Learning_P_Gamma_"+g[1]+"_epsilon_05_alpha_0.1.png")
        plochange = []
        for i in range(len(V_track)):
            plochange.append(V_track[i]/ 64)
        plot(plochange, "frozen_lake_Q_Learning_mean_state_values_gamma_"+g[1]+"_epsilon_05_alpha_0.1", "mean state value")

        V_V.append(plochange)

        plochange = []
        for i in range(len(pi_track)):
            mat = np.abs(pi_track[i]-pi_track[-1])
            mat_edit = mat
            mat_edit[mat > 1] = 1
            plochange.append(np.sum(mat_edit))
        #plot(plochange, "frozen_lake_Q_Learning_policy_gamma_1", "number of varied states from converged policy")

        plochange_t = []
        for i in range(len(pi_track)):
            mat = np.abs(pi_track[i] - np.array(list(pi_track_VI[-1].values())))
            mat_edit = mat
            mat_edit[mat > 1] = 1
            plochange_t.append(np.sum(mat_edit))
        #plot(plochange_t, "frozen_lake_Q_Learning_true_policy_gamma_1", "number of varied states from converged to true policy")
        plot3(plochange,plochange_t, "frozen_lake_Q_Learning_policy_gamma_"+g[1]+"_epsilon_05_alpha_0.1", "number of varied states from converged(true) policy")

        P_P.append(plochange)
        P_P_t.append(plochange_t)

    plot5(V_V, "frozen_lake_Q_Learning_mean_state_values_convergence","mean state value")
    plot5(P_P, "frozen_lake_Q_Learning_convergence_to_final_policy", "number of varied states from converged policy")
    plot5(P_P_t, "frozen_lake_Q_Learning_convergence_to_true_policy", "number of varied states from true policy")



frozen_lake = gym.make('Blackjack-v1', render_mode=None)
base_env = gym.make('Blackjack-v1', render_mode=None)
frozen_lake = BlackjackWrapper(base_env)

if 1:
    for alg in ("VI","PI"):
        print("--------------------")
        print(alg+" data bellow")
        print("--------------------")
        xx=[1,0.9,0.5,0.1]
        v_t=[]
        v_i=[]
        p_t=[]
        p_i=[]
        v_v=[]
        for g in [(1,"1"),(0.9,"09"),(0.5,"05"),(0.1,"01")]:
            base_env = gym.make('Blackjack-v1',sab=True, render_mode=None)
            frozen_lake = BlackjackWrapper(base_env)
            print("--------------------")
            if alg=="VI":
                V, V_track, pi, pi_track, iter_stop, pi_stop, t_con = Planner(frozen_lake.P).value_iteration(gamma=g[0], n_iters=50,theta=1e-20)
            else:
                V, V_track, pi, pi_track, iter_stop, pi_stop, t_con = Planner(frozen_lake.P).policy_iteration(gamma=g[0],n_iters=20,seed=812,theta=1e-20)
            v_t.append(t_con)
            v_i.append(iter_stop)
            p_i.append(pi_stop)
            v_v.append(np.sum(V)/290)
            print(t_con, "seconds to converge "+alg+" Black Jack at gamma="+g[1]+", and standard epsilon")
            print(iter_stop, "iteration to converge states values to standard epsilon")
            print(pi_stop, "iteration to converge to stable policy")

            plochange = []
            for i in range(V_track.shape[0]):
                plochange.append(np.sum(V_track[i]) / 290)
            plot(plochange, "black_jack_"+alg+"_mean_state_values_gamma_"+g[1], "mean state value")

            plochange = []
            for i in range(len(pi_track)):
                mat = np.abs(np.array(list(pi_track[i].values())) - np.array(list(pi_track[-1].values())))
                mat_edit = mat
                mat_edit[mat > 1] = 1
                plochange.append(np.sum(mat_edit))
            plot(plochange, "black_jack_"+alg+"_policy_gamma_"+g[1], "number of varied states from converged policy")

            if alg == "Vi":
                V, V_track, pi, pi_track, iter_stop, pi_stop, t_con = Planner(frozen_lake.P).value_iteration(gamma=g[0], n_iters=pi_stop,theta=1e-20)
            else:
                V, V_track, pi, pi_track, iter_stop, pi_stop, t_con = Planner(frozen_lake.P).policy_iteration(gamma=g[0],n_iters=pi_stop,seed=812,theta=1e-20)
            p_t.append(t_con)
            print(t_con, "seconds to converge "+alg+" black jack at gamma="+g[1]+", and standard epsilon until stable policy")
        plot2(xx, v_t, "black_jack_" + alg + "_value_convergence_time", "seconds to converged converged values","gamma")
        plot2(xx, p_t, "black_jack_" + alg + "_policy_convergence_time", "seconds to converged converged policy","gamma")
        plot2(xx, v_i, "black_jack_" + alg + "_value_convergence_iterations", "iterations to converged converged values","gamma")
        plot2(xx, p_i, "black_jack_" + alg + "_policy_convergence_iterations", "iterations to converged converged policy","gamma")
        plot2(xx, v_v, "black_jack_" + alg + "_mean_state_value","mean state value", "gamma")

if 1:
    iters=100000

    V_VI, V_track_VI, pi_VI, pi_track_VI, iter_stop_VI, pi_stop_VI, t_con_VI = Planner(frozen_lake.P).value_iteration(gamma=1, n_iters=500,theta=1e-20)
    for e in ((1,"1"), (0.3,"03"), (0.1,"01")):
        Q, V, pi, Q_track, pi_track, V_track = RL(frozen_lake).q_learning(gamma=1,init_alpha=0.1,min_alpha=0.1,init_epsilon=e[0],min_epsilon=e[0], n_episodes=iters,seed=812)
        mat = np.abs(pi_track[-1] - np.array(list(pi_track_VI[-1].values())))
        mat_edit = mat
        mat_edit[mat > 1] = 1
        print(np.sum(mat_edit),"policy difference in between Q Learning and VI, gamma=1,epsilon="+e[1]+" alpha=0.1")
        plochange = []
        for i in range(len(V_track)):
            plochange.append(V_track[i]/ 64)
        plot(plochange, "black_jack_Q_Learning_mean_state_values_gamma_1_epsilon_"+e[1]+"_alpha_01", "mean state value")

        plochange = []
        for i in range(len(pi_track)):
            mat = np.abs(pi_track[i]-pi_track[-1])
            mat_edit = mat
            mat_edit[mat > 1] = 1
            plochange.append(np.sum(mat_edit))
        #plot(plochange, "frozen_lake_Q_Learning_policy_gamma_1", "number of varied states from converged policy")

        plochange_t = []
        for i in range(len(pi_track)):
            mat = np.abs(pi_track[i] - np.array(list(pi_track_VI[-1].values())))
            mat_edit = mat
            mat_edit[mat > 1] = 1
            plochange_t.append(np.sum(mat_edit))
        #plot(plochange_t, "frozen_lake_Q_Learning_true_policy_gamma_1", "number of varied states from converged to true policy")
        plot3(plochange,plochange_t, "black_jack_Q_Learning_policy_gamma_1_epsilon_"+e[1]+"_alpha_01", "number of varied states from converged(true) policy")

if 1:
    iters=100000

    V_VI, V_track_VI, pi_VI, pi_track_VI, iter_stop_VI, pi_stop_VI, t_con_VI = Planner(frozen_lake.P).value_iteration(gamma=1, n_iters=500)
    for a in ((0.3,"03"), (0.03,"003")):
        Q, V, pi, Q_track, pi_track, V_track = RL(frozen_lake).q_learning(gamma=1,init_alpha=a[0],min_alpha=a[0],init_epsilon=0.3,min_epsilon=0.3, n_episodes=iters,seed=812)
        mat = np.abs(pi_track[-1] - np.array(list(pi_track_VI[-1].values())))
        mat_edit = mat
        mat_edit[mat > 1] = 1
        print(np.sum(mat_edit),"policy difference in between Q Learning and VI, gamma=1,epsilon=0.3 alpha="+a[1])
        plochange = []
        for i in range(len(V_track)):
            plochange.append(V_track[i]/ 64)
        plot(plochange, "black_jack_Q_Learning_mean_state_values_gamma_1_epsilon_03_alpha_"+a[1], "mean state value")

        plochange = []
        for i in range(len(pi_track)):
            mat = np.abs(pi_track[i]-pi_track[-1])
            mat_edit = mat
            mat_edit[mat > 1] = 1
            plochange.append(np.sum(mat_edit))
        #plot(plochange, "frozen_lake_Q_Learning_policy_gamma_1", "number of varied states from converged policy")

        plochange_t = []
        for i in range(len(pi_track)):
            mat = np.abs(pi_track[i] - np.array(list(pi_track_VI[-1].values())))
            mat_edit = mat
            mat_edit[mat > 1] = 1
            plochange_t.append(np.sum(mat_edit))
        #plot(plochange_t, "frozen_lake_Q_Learning_true_policy_gamma_1", "number of varied states from converged to true policy")
        plot3(plochange,plochange_t, "black_jack_Q_Learning_policy_gamma_1_epsilon_03_alpha_"+a[1], "number of varied states from converged(true) policy")

if 1:
    iters=100000

    for g in ((0.5,"05"), (0.1,"01")):
        V_VI, V_track_VI, pi_VI, pi_track_VI, iter_stop_VI, pi_stop_VI, t_con_VI = Planner(frozen_lake.P).value_iteration(gamma=g[0], n_iters=500)
        Q, V, pi, Q_track, pi_track, V_track = RL(frozen_lake).q_learning(gamma=g[0],init_alpha=0.1,min_alpha=0.1,init_epsilon=0.3,min_epsilon=0.3, n_episodes=iters,seed=812)
        mat = np.abs(pi_track[-1] - np.array(list(pi_track_VI[-1].values())))
        mat_edit = mat
        mat_edit[mat > 1] = 1
        print(np.sum(mat_edit),"policy difference in between Q Learning and VI, gamma="+g[1]+",epsilon=0.3 alpha=0.1")
        plochange = []
        for i in range(len(V_track)):
            plochange.append(V_track[i]/ 64)
        plot(plochange, "black_jack_Q_Learning_mean_state_values_gamma_"+g[1]+"_epsilon_03_alpha_0.1", "mean state value")

        plochange = []
        for i in range(len(pi_track)):
            mat = np.abs(pi_track[i]-pi_track[-1])
            mat_edit = mat
            mat_edit[mat > 1] = 1
            plochange.append(np.sum(mat_edit))
        #plot(plochange, "frozen_lake_Q_Learning_policy_gamma_1", "number of varied states from converged policy")

        plochange_t = []
        for i in range(len(pi_track)):
            mat = np.abs(pi_track[i] - np.array(list(pi_track_VI[-1].values())))
            mat_edit = mat
            mat_edit[mat > 1] = 1
            plochange_t.append(np.sum(mat_edit))
        #plot(plochange_t, "frozen_lake_Q_Learning_true_policy_gamma_1", "number of varied states from converged to true policy")
        plot3(plochange,plochange_t, "black_jack_Q_Learning_policy_gamma_"+g[1]+"_epsilon_05_alpha_0.1", "number of varied states from converged(true) policy")









if 0:
    V_VI, V_track_VI, pi_VI, pi_track_VI, iter_stop_VI, pi_stop_VI, t_con_VI = Planner(frozen_lake.P).value_iteration(gamma=1, n_iters=500,theta=1e-20)
    #print(V_VI.reshape(10,29))
    size=(29,10)

    Plots.values_heat_map(V_VI, "Frozen Lake Q_Learning\nState Values gamma=", size,"test.png")
    size=(10,29)
    Plots.values_heat_map(V_VI, "Frozen Lake Q_Learning\nState Values gamma=", size,"test2.png")

    Q, V, pi, Q_track, pi_track, V_track = RL(frozen_lake).q_learning(gamma=1,init_alpha=0.1,min_alpha=0.1,init_epsilon=1,min_epsilon=1, n_episodes=10000,seed=812)
    size=(29,10)
    Plots.values_heat_map(V, "Frozen Lake Q_Learning\nState Values gamma=", size,"test3.png")



if 0:

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