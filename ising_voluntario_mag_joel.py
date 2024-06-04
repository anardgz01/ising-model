import numpy as np
import ising_simulator_voluntario as ising
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import matrix_reshaper
# from numba import njit, prange

N = np.array([16, 32, 64, 128])
T = np.linspace(1.5, 3.5, 10)
T = np.sort(T)
mags_exps = np.zeros((4, len(T)*len(N)))
energy_avgs = np.zeros((4, len(T)*len(N)))
energy_avgs_normalized = np.zeros((4, len(T)*len(N)))
specific_heat_avgs = np.zeros((4, len(T)*len(N)))

# def expected_value(file_mags : str):
#         mags = np.load(file_mags)
#         print(f'len mags es {len(mags)}')
        
#         hist, bin_edges = np.histogram(mags, bins=10000, range=(0,1.0001))
#         hist = hist/len(mags)
#         print(f'len hist es {len(hist)}')
#         print(f'mags is {mags}')
#         bin_indices = np.digitize(mags, bin_edges) -1
#         print(f'bin_indices es {bin_indices}')
#         print(f'bin_edges es {bin_edges}')
        
#         ex_value = 0
#         counted_indices = set()
#         for mag_index, mag in enumerate(mags):
#             index = bin_indices[mag_index]
#             if index in counted_indices:
#                 continue
#             counted_indices.add(index)
#             mag *= hist[index]
#             ex_value += mag
#         return ex_value

def data_average_magnetization(file_mags : str):
    mags = np.load(file_mags)
    std_mags = np.std(mags) / np.sqrt(len(mags))
    return np.mean(mags), std_mags

def data_mean_energy(N, file_energies : str):
    '''Returns a pair of pairs, where the first is normalized by 2N and the second is normalized by 2N^2.'''
    energies = np.load(file_energies)
    std_energies = np.std(energies) / np.sqrt(len(energies))
    return (np.mean(energies)/(2*N), std_energies/(2*N)), (np.mean(energies)/(2*N**2), std_energies/(2*N**2))

def data_average_specific_heat(N, t, file_energies : str):
    energies = np.load(file_energies)
    print(f'len energies es {len(energies)}, min energies es {np.min(energies)}, max energies es {np.max(energies)}')
    mean_energies = np.mean(energies)
    mean_energies_sq = np.mean(energies**2)
    std_energies = np.std(energies)  # Compute standard error of the mean
    std_energies_sq = np.std(energies**2)   # Compute standard error of the mean
    print(f'error energies is {std_energies}, error energies sq is {std_energies_sq}')
    return (mean_energies_sq - mean_energies**2)/(N**2*t), std_energies_sq/(N**2*t) + 2*std_energies*mean_energies/(N**2*t) #+ 2 * std_energies_sq * std_energies * cov_energies/(N**4 * t**2))
    # return (mean_energies_sq - mean_energies**2)/(N**2*t), np.sqrt((std_energies_sq/(N**2*t))**2 + (2*std_energies*mean_energies/(N**2*t))**2) #+ 2 * std_energies_sq * std_energies * cov_energies/(N**4 * t**2))
    # return (np.mean(energies**2)-np.mean(energies)**2)/(N**2*t), np.sqrt((np.std(energies**2)/(N**2*t))**2+(np.std(energies)*2*np.mean(energies)/(N**2*t))**2)

# for index, t in enumerate(T):
#     path = f'temp_{t:.2f}'
#     ising.simulate(True, t, N, 1000000, path)
#     mags_exps[0,index] = t
#     mags_exps[1,index] = expected_value(f'resultados/mags_{path}.npy')
#     mags_exps[2,index] = data_average_magnetization(f'resultados/mags_{path}.npy')
#     print(f'finished simulation {index+1} of {len(T)}')

def run_simulation(N_temp_pair : tuple[int, float]):
    try:
        n = N_temp_pair[0]
        t = N_temp_pair[1]
        path = f'N_{n}_temp_{t:.2f}'

        # if t < 1:
        #     iterations = 100000
        # elif t < 2:
        #     iterations = 10000
        # else:
        #     iterations = 3000
        iterations = 1000000

        ising.simulate(True, t, n, iterations, path)
        print(f'finished simulation with temp {t} and N {n}')
    except Exception as e:
        print(f"Exception: {e}")

def calculate_data(temp_N_pair : tuple[float, int], index):
    global mags_exps, energy_avgs, specific_heat_avgs
    t = temp_N_pair[0]
    n = temp_N_pair[1]
    path = f'N_{n}_temp_{t:.2f}'

    mags_exps[0,index] = energy_avgs[0,index] = specific_heat_avgs[0,index] = energy_avgs_normalized[0,index] = n
    mags_exps[1,index] = energy_avgs[1,index] = specific_heat_avgs[1,index] = energy_avgs_normalized[1,index] = t
    mags_exps[2,index], mags_exps[3,index] = data_average_magnetization(f'resultados/mags_{path}.npy')
    energy_avgs[2,index], energy_avgs[3,index] = data_mean_energy(n, f'resultados/energies_{path}.npy')[0]
    energy_avgs_normalized[2,index], energy_avgs_normalized[3,index] = data_mean_energy(n, f'resultados/energies_{path}.npy')[1]
    specific_heat_avgs[2,index], specific_heat_avgs[3, index] = data_average_specific_heat(n, t, f'resultados/energies_{path}.npy')
    print(f'finished simulation {index+1} of {len(T)*len(N)} with temp {t} and N {n}')

prod = product(N, T)
# prod_list = list(prod)        # Use these lines to remove specific elements before running
# prod_list = [(x,y) for x,y in prod_list if not (x==16 and y < 2.39)]
# print(prod_list)
# prod = iter(prod_list)

with ProcessPoolExecutor(max_workers=10) as executor:
    executor.map(run_simulation, prod)

for index, t in enumerate(T):
    for j, n in enumerate(N):
        calculate_data((t, n), index*len(N)+j)

# @njit(parallel=True)
# def parallel_function():
#     for index, t in enumerate(T):
#         path = f'temp_{t:.2f}'
#         ising.simulate(True, t, N, 1000000, path)
#         mags_exps[0,index] = energy_avgs[0,index] = specific_heat_avgs[0,index] = t
#         mags_exps[1,index] = data_average_magnetization(f'resultados/mags_{path}.npy')
#         energy_avgs[1,index] = data_mean_energy(n, f'resultados/energies_{path}.npy')
#         specific_heat_avgs[1,index] = data_average_specific_heat(n, t, f'resultados/energies_{path}.npy')
#         print(f'finished simulation {index+1} of {len(T)*len(N)} with temp {t} and N {n} ({(index+1)/(len(T)*len(N))*100:.2f}% completed')

np.save(f'resultados/mags_exps.npy', mags_exps)
np.save(f'resultados/energies_avgs.npy', energy_avgs)
np.save(f'resultados/energies_avgs_normalized.npy', energy_avgs_normalized)
np.save(f'resultados/specific_heat_avgs.npy', specific_heat_avgs)

matrix_reshaper.reshape()
