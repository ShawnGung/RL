import numpy as np
from env import *
V = np.zeros(nS)

def best_value(s,V):
    Vk_list = np.zeros(nA)
    for a in range(nA):
        for prob,next_state,reward,done in ENV[s][a]:
            Vk_list[a]+=prob*(reward + gama*V[next_state])
    return Vk_list

def value_iteration():
    while True:
        delta = 0
        for s in range(nS):
            best_action_value = max(best_value(s, V))
            delta = max(delta, np.abs(best_action_value - V[s]))
            V[s] = best_action_value

        if delta < theta:
            break

    policy = np.zeros([nS, nA])
    for s in range(nS):
        A = best_value(s, V)
        best_action = np.argmax(A)
        policy[s, best_action] = 1.0
    return policy


print("Policy Probability Distribution:")
print(value_iteration())
print("")
