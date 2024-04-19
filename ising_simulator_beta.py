'''Realizar el programa que simular el Modelo de Ising con el método Monte Carlo. Mostrar su evolución por
pantalla para diversas temperaturas.'''

import numpy as np
import argparse

def simulate(sorted : bool, T : float, N : int, num_Monte_Carlo_steps : int, path : str):
    Monte_Carlo_step = N**2
    k_boltzman = 1.380649e-23
    iterations = num_Monte_Carlo_steps*Monte_Carlo_step
    # confs = np.zeros((iterations,N,N), dtype=int)
    magnetizations_mcs = np.zeros((num_Monte_Carlo_steps//100+1))
    factor_conf_mcs = np.zeros((num_Monte_Carlo_steps//100+1))

    #Create the initial state
    if sorted:
        conf = np.full((N,N), 1)
        print('sorted')
    else:
        conf = np.random.choice([1, -1], size=(N, N))
        print('unsorted')
    # confs[0] = conf

    #print(confs[0])

    #Functions
    def get_neighbourhood(pos : tuple[int,int]):
        '''Take the chosen position and return an array of the positions of its nearest neighbours (self, up, down, right, left)'''
        positions = [(0,0)]*5
        positions [0] = pos
        positions [1] = ((pos[0]+1) % N, pos[1])
        positions [2] = ((pos[0]-1) % N, pos[1])
        positions [3] = (pos[0], (pos[1]+1) % N)
        positions [4] = (pos[0], (pos[1]-1) % N)
        return positions

    def delta_E(pos):
        '''Return the energy of the system formed by the given electron and its neighbours'''
        positions = get_neighbourhood(pos)
        energy = 2 * conf[pos] * (conf[positions[1]] + conf[positions[2]] + conf[positions[3]] + conf[positions[4]])
        return energy

    def p(pos, T=T):
        '''Return the probability of the transition to a new state with opposite spin'''
        return min(1, np.e**(-delta_E(pos)/T))

    def magnetization(conf):
        '''Return the magnetization of the system'''
        return np.sum(conf)/N**2
    
    def energy_conf(conf):
        energy = 0.
        for i in range(N):
            for j in range(N):
                pos_nn = get_neighbourhood((i,j))
                energy += -0.5*conf[pos_nn[0]] * (conf[pos_nn[1]] + conf[pos_nn[2]] + conf[pos_nn[3]] + conf[pos_nn[4]])
        return energy
    
    def factor_conf(conf):
        energy = energy_conf(conf)
        print(energy)
        factor = np.e**(-energy/T)
        print(factor)
        return factor


    landmark = iterations // 20 #print progress every 5%
    #Start iterating
    for t in range(1,iterations+1):
        pos = (np.random.randint(0, N), np.random.randint(0, N))
        probability = p(pos)
        ksi = np.random.uniform(0.,1.)

        if ksi < probability:
            conf[pos] *= -1

        # confs[t] = conf
        
        if t % (Monte_Carlo_step*100) == 0:
            magnetizations_mcs[t//(Monte_Carlo_step*100)] = magnetization(conf)
            factor_conf_mcs[t//(Monte_Carlo_step*100)] = factor_conf(conf)

        if t % landmark == 0:
            print(f'Iteration number {t+1} of {iterations} ({(t // landmark)*5}% completed)')

    probability_mcs = factor_conf_mcs/np.sum(factor_conf_mcs)

    np.save(f'resultados/mags_{path}.npy', magnetizations_mcs)
    np.save(f'resultados/probs_{path}.npy', probability_mcs)

