#Realizar el programa que simular el Modelo de Ising con el método Monte Carlo. Mostrar su evolución por
#pantalla para diversas temperaturas.

import numpy as np
import matplotlib.pyplot as plt

#Parameters
N = 2
num_Monte_Carlo_steps = 10
Monte_Carlo_step = N**2
iterations = num_Monte_Carlo_steps*Monte_Carlo_step
T = 4
confs = np.zeros((iterations,N,N))
confs_mcs = np.zeros((num_Monte_Carlo_steps+1,N,N))

#Create the initial state
conf = np.random.choice([1, -1], size=(N, N))
confs[0] = conf
confs_mcs[0] = conf

#print(confs[0])

#Functions
def get_neighbourhood(pos):
    global N
    positions = np.zeros(5)
    positions [0] = pos
    positions [1] = (pos[0]+1 % N, pos[1])
    positions [2] = (pos[0]-1 % N, pos[1])
    positions [3] = (pos[0], pos[1]+1 % N)
    positions [4] = (pos[0], pos[1]-1 % N)
    return positions

def delta_E (pos):
    global conf
    positions = get_neighbourhood(pos)
    energy = 2 * conf[pos] * (conf[positions[1]] + conf[positions[2]] + conf[positions[3]] + conf[positions[4]])
    return energy

def p(pos, T=T):
    return np.min(1, np.e**(-delta_E(pos)/T))

#Start iterating
for t in range(1,iterations):
    pos = (np.random.randint(0, N), np.random.randint(0, N))
    probability = p(t, pos)
    psi = np.random.uniform(0.,1.)

    if psi < probability:
        conf[pos] *= -1

    confs[t] = conf
    
    if t % Monte_Carlo_step == 0:
        confs_mcs = confs[t]
confs_mcs[-1] = conf
