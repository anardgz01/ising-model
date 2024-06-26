'''Realizar el programa que simular el Modelo de Ising con el método Monte Carlo. Mostrar su evolución por
pantalla para diversas temperaturas.'''

import numpy as np
import time
import cProfile

def simulate(sorted : bool, T : float, N : int, num_Monte_Carlo_steps : int, path : str):
    #initial parameters
    percent = 2
    Monte_Carlo_step = N**2
    iterations = num_Monte_Carlo_steps*Monte_Carlo_step
    display_landmark = iterations // (100//percent) #print progress every {percent}%
    # measuring_landmark = 10 if num_Monte_Carlo_steps <= 10000 else 100
    measuring_landmark = 100

    #initialize parameters
    magnetizations_mcs = np.zeros((num_Monte_Carlo_steps//measuring_landmark))
    energies_mcs = np.zeros((num_Monte_Carlo_steps//measuring_landmark))
    correlations_mcs = np.zeros((num_Monte_Carlo_steps//measuring_landmark, N//2))

    #Create the initial state
    if sorted:
        conf = np.full((N,N), 1)
        print('sorted')
    else:
        conf = np.random.choice([1, -1], size=(N, N))
        print('unsorted')

    #Functions
    # def get_neighbourhood(pos : tuple[int,int]):  #old, slow version
    #     '''Take the chosen position and return an array of the positions of its nearest neighbours (self, up, down, right, left)'''
    #     positions = [(0,0)]*5
    #     positions [0] = pos
    #     positions [1] = ((pos[0]+1) % N, pos[1])
    #     positions [2] = ((pos[0]-1) % N, pos[1])
    #     positions [3] = (pos[0], (pos[1]+1) % N)
    #     positions [4] = (pos[0], (pos[1]-1) % N)
    #     return positions
    def get_neighbourhood(x : int, y : int):
        '''Take the chosen position and return an array of the positions of its nearest neighbours (self, up, down, right, left)'''
        return [
            (x, y),
            ((x+1) % N, y),
            ((x-1) % N, y),
            (x, (y+1) % N),
            (x, (y-1) % N)
        ]

    def delta_E(x : int, y : int):
        '''Return the energy of the system formed by the given electron and its neighbours'''
        positions = get_neighbourhood(x, y)
        energy_increment = 2 * conf[positions[0]] * (conf[positions[1]] + conf[positions[2]] + conf[positions[3]] + conf[positions[4]])
        return energy_increment

    def initial_p_calcs():
        '''Return the probabilities of the transitions to a new state with opposite spin. Only called once to define the list'''
        possible_probabilities = {
        8 : min(1, np.e**(8/T)),
        4 : min(1, np.e**(4/T)),
        0 : min(1, np.e**(0/T)),
        -4 : min(1, np.e**(-4/T)),
        -8 : min(1, np.e**(-8/T))
        }
        return possible_probabilities
        
    # def p_2(pos):
    #     '''Return the probability of the transition to a new state with opposite spin'''
    #     return min(1, np.e**(-delta_E(pos)/T))
    
    def p(x : int, y : int):
        '''Return the probability of the transition to a new state with opposite spin'''
        return possible_probabilities[delta_E(x,y)]

    def magnetization(conf):
        '''Return the magnetization of the system'''
        return np.abs(np.sum(conf)/N**2)
    
    # def energy_conf(conf):    #old, slow version
    #     energy = 0.
    #     for i in range(N):
    #         for j in range(N):
    #             pos_nn = get_neighbourhood((i,j))
    #             energy += -0.5*conf[pos_nn[0]] * (conf[pos_nn[1]] + conf[pos_nn[2]] + conf[pos_nn[3]] + conf[pos_nn[4]])
    #     return energy
    
    def energy_conf(conf): #vectorized version, which should be faster
        '''Return the energy of the system'''
        conf_up = np.roll(conf, 1, axis=0)  #s(m+1,n) in conf is s(m, n) here.
        conf_down = np.roll(conf, -1, axis=0)
        conf_right = np.roll(conf, 1, axis=1)
        conf_left = np.roll(conf, -1, axis=1)

        neighbor_sum = conf_up + conf_down + conf_right + conf_left
        energy = -0.5 * conf * neighbor_sum

        return np.sum(energy)
    
    # def correlation_function_old(conf, distance):
    #     correlation = np.zeros(N**2)
    #     index = 0
    #     for n in range(N):
    #         for m in range(N):
    #             correlation[index] = conf[n, m] * conf[(n+distance)%N, m]
    #             index +=1
    #     return np.sum(correlation)/N**2
    
    def correlation_function(conf, distance):
        shifted_conf = np.roll(conf, shift=-distance, axis=1)   #this creates a new matrix by rotating places, such as s(m+distance,n) in conf is s(m, n) in shifted_conf.
        correlation = conf * shifted_conf   #this is a matrix where x(m,n) is the product of conf[m, n] and conf[m+distance, n]. 
        return np.sum(correlation)/N**2
    
    # def correlation_function_global(distance, confs_array):
    #     correlations = np.zeros(len(confs_array))

    #     for i, conf in enumerate(confs_array):
    #         correlations[i] = correlation_function(conf, distance)
    #     return np.mean(correlations)

    start_time = time.time()
    possible_probabilities = initial_p_calcs()
    
    def generate_random_values(min_landmark):
        random_positions_x = np.random.randint(0, N, size=min_landmark, dtype=np.uint8)
        random_positions_y = np.random.randint(0, N, size=min_landmark, dtype=np.uint8)
        ksi_values = np.random.uniform(0.,1., size=min_landmark)
        return random_positions_x, random_positions_y, ksi_values
    
    #Start iterating
    measuring_index = -1
    stop = False
    last_measure_index = len(magnetizations_mcs)-1
    measuring_landmark_iterations = measuring_landmark*Monte_Carlo_step
    min_random_array_size = 10000000     #minimum size of the random values array. Should increase speed and memory usage
    min_landmark = (min_random_array_size // measuring_landmark_iterations + 1) * measuring_landmark_iterations

    while True:
        if stop:
            break
        random_positions_x, random_positions_y, ksi_values = generate_random_values(min_landmark)
        for t in range(1,min_landmark+1):
            x = random_positions_x[t-1]
            y = random_positions_y[t-1]
            probability = p(x, y)
            # ksi = np.random.uniform(0.,1.)

            if ksi_values[t-1] < probability:
                conf[x,y] *= -1
            
            if t % measuring_landmark_iterations == 0:
                measuring_index += 1
                print(f'measuring at {t} of {min_landmark}, with index {measuring_index} of {last_measure_index}')
                # measure_index = t//(Monte_Carlo_step*measuring_landmark)-1  #NOTE THE -1
                magnetizations_mcs[measuring_index] = magnetization(conf)
                energies_mcs[measuring_index] = energy_conf(conf)
                for i in range(N//2):
                    correlations_mcs[measuring_index, i] = correlation_function(conf, i+1)
                if measuring_index == last_measure_index:
                    stop = True
                    break

    print(f'simulation finished in {time.time() - start_time}')

    # correlation_data = np.array((np.mean(correlations_mcs, axis=0), np.std(correlations_mcs, axis=0)))
    
    # np.save(f'resultados/mags_{path}.npy', magnetizations_mcs)
    # np.save(f'resultados/energies_{path}.npy', energies_mcs)
    # np.save(f'resultados/correlations_global_{path}.npy', correlation_data)
cProfile.run('simulate(True, 2.3, 32, 5000, "test")')
