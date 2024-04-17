'''Realizar el programa que simular el Modelo de Ising con el método Monte Carlo. Mostrar su evolución por
pantalla para diversas temperaturas.'''

import numpy as np
import plot_animation_ising as my_plot
import argparse

def simulate(sorted : bool, T : float, N : int, num_Monte_Carlo_steps : int, path : str):
    Monte_Carlo_step = N**2
    iterations = num_Monte_Carlo_steps*Monte_Carlo_step
    # confs = np.zeros((iterations,N,N), dtype=int)
    confs_mcs = np.zeros((num_Monte_Carlo_steps+1,N,N), dtype=np.int8)
    magnetizations_mcs = np.zeros((num_Monte_Carlo_steps+1))

    #Create the initial state
    if sorted:
        conf = np.full((N,N), 1)
        print('sorted')
    else:
        conf = np.random.choice([1, -1], size=(N, N))
        print('unsorted')
    # confs[0] = conf
    confs_mcs[0] = conf

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

    landmark = iterations // 20 #print progress every 5%
    #Start iterating
    for t in range(1,iterations+1):
        pos = (np.random.randint(0, N), np.random.randint(0, N))
        probability = p(pos)
        ksi = np.random.uniform(0.,1.)

        if ksi < probability:
            conf[pos] *= -1

        # confs[t] = conf
        
        if t % Monte_Carlo_step == 0:
            confs_mcs[t//Monte_Carlo_step] = conf
            magnetizations_mcs[t//Monte_Carlo_step] = magnetization(conf)

        if t % landmark == 0:
            print(f'Iteration number {t+1} of {iterations} ({(t // landmark)*5}% completed)')

    np.save(f'resultados/confs_{path}.npy', confs_mcs)
    np.save(f'resultados/mags_{path}.npy', magnetizations_mcs)

if __name__ == '__main__':
# Create the parser and add the -s argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--sorted', action='store_true', help='Start with an ordered lattice')
    parser.add_argument('-t', '--temperature', type=float, default=1, action='store', help='Specify the temperature in Kelvin degrees')
    parser.add_argument('-n', type=int, default=32, action='store', help='Specify the number of electrons per side of the square lattice')
    parser.add_argument('-s', '--steps', type=int, default=100, action='store', help='Specify the number of Monte Carlo steps to be simulated')
    parser.add_argument('-o', '--output', type=str, default='', action='store', help='Specify the string to be appended to the saved arrays')
    args = parser.parse_args()

    #Parameters
    sorted = args.sorted
    N = args.n
    T = args.temperature
    num_Monte_Carlo_steps = args.steps
    path = args.output

    simulate(sorted, T, N, num_Monte_Carlo_steps, path)
    my_plot.plot()
