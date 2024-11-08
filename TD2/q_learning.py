import random as rd
import gym
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
import time


def update_q_table(Q, s, a, r, sprime, alpha, gamma):
    """
    This function should update the Q function for a given pair of action-state
    following the q-learning algorithm, it takes as input the Q function, the pair action-state,
    the reward, the next state sprime, alpha the learning rate and gamma the discount factor.
    Return the same input Q but updated for the pair s and a.
    """
    #recherche du maximum de Q[sprime][aprime]-Q[s][a]
    aprime = np.argmax(Q[sprime])

    #application de la fonction
    Q[s][a] = Q[s][a]+alpha*(r+gamma*Q[sprime][aprime]-Q[s][a])
    return Q


def epsilon_greedy(Q, s, epsilone):
    """
    This function implements the epsilon greedy algorithm.
    Takes as unput the Q function for all states, a state s, and epsilon.
    It should return the action to take following the epsilon greedy algorithm.
    """

    randint=rd.randint(0,5)
    random_number = rd.random()
    if random_number<epsilone:
        a=randint
    else:
        a=np.argmax(Q[s])
    return a

def eps(b):
    Y=[]
    for x in range(b):
        Y.append(np.exp(-x/30))
    return Y

if __name__ == "__main__":
    env = gym.make("Taxi-v3", render_mode="human")

    env.reset()
    env.render()

    Q = np.zeros([env.observation_space.n, env.action_space.n])

    alpha = 0.1 # choose your own

    gamma = 0.8 # choose your own

    

    n_epochs = 1000 # choose your own
    max_itr_per_epoch = 1000 # choose your own
    rewards = []
    episodes=[]

    Epsilon = eps(n_epochs) # choose your own
    for e in range(n_epochs):
        r = 0
        epsilon = Epsilon[e]
        S, _ = env.reset()
        moves = 0

        for i in range(max_itr_per_epoch):
            A = epsilon_greedy(Q=Q, s=S, epsilone=epsilon)

            Sprime, R, done, _, info = env.step(A)

            r += R

            Q = update_q_table(
                Q=Q, s=S, a=A, r=R, sprime=Sprime, alpha=alpha, gamma=gamma
            )
            moves = i
            # Update state and put a stoping criteria
            S=Sprime
            if done:
                break


        print("episode #", e, " : r = ", r, "Nombre de coups : ", moves ,"epsilon = ", epsilon)
        episodes.append(e)
        rewards.append(r)

    print("Average reward = ", np.mean(rewards))
    plt.figure()
    plt.plot(episodes,rewards)
    plt.show()

    # plot the rewards in function of epochs

    print("Training finished.\n")

    
    """
    
    Evaluate the q-learning algorihtm
    
    """

    env.close()
