import numpy as np
import ising_simulator_beta as ising
# import os

N = 32
T = np.linspace(0.01, 5, 10)
T = np.sort(T)
mags_exps = np.zeros((3, len(T)))

# Delete all contents in 'resultados' folder
# folder_path = '/home/anardgz01/Documentos/GitHub/ising-model/resultados'
# for filename in os.listdir(folder_path):
#     file_path = os.path.join(folder_path, filename)
#     if os.path.isfile(file_path):
#         os.unlink(file_path)

def expected_value(file_mags : str):
        mags = np.load(file_mags)
        print(f'len mags es {len(mags)}')
        
        hist, bin_edges = np.histogram(mags, bins=10000, range=(0,1.0001))
        # bin_edges[-1] = 1.01
        hist = hist/len(mags)
        print(f'len hist es {len(hist)}')
        print(f'mags is {mags}')
        bin_indices = np.digitize(mags, bin_edges) -1
        print(f'bin_indices es {bin_indices}')
        print(f'bin_edges es {bin_edges}')
        
        ex_value = 0
        counted_indices = set()
        for mag_index, mag in enumerate(mags):
            index = bin_indices[mag_index]
            if index in counted_indices:
                continue
            counted_indices.add(index)
            mag *= hist[index]
            ex_value += mag
        return ex_value

def average_magnetization(file_mags : str):
    mags = np.load(file_mags)
    return np.mean(mags)

# def average_magnetizations(file_mags : str, file_probs : str) -> tuple[float, float]:
#         mags = np.load(file_mags)
#         probs = np.load(file_probs)
#         mags_avgs = np.sum(probs*mags)
#         return mags_avgs

for index, t in enumerate(T):
    path = f'temp_{t:.2f}'
    ising.simulate(True, t, N, 1000000, path)
    mags_exps[0,index] = t
    mags_exps[1,index] = expected_value(f'resultados/mags_{path}.npy')
    mags_exps[2,index] = average_magnetization(f'resultados/mags_{path}.npy')
    print(f'finished simulation {index+1} of {len(T)}')

np.save(f'resultados/mags_exps.npy', mags_exps)