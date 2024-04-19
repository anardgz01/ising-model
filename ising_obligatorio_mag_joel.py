import numpy as np
import ising_simulator_beta as ising
import os

N = 32
T = np.random.uniform(0.01, 5, 10)
T = np.sort(T)

# Delete all contents in 'resultados' folder
folder_path = '/home/anardgz01/Documentos/GitHub/ising-model/resultados'
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    if os.path.isfile(file_path):
        os.unlink(file_path)

def average_magnetization(file_mags : str, file_probs : str) -> tuple[float, float]:
        mags = np.load(file_mags)
        probs = np.load(file_probs)
        mags_avgs = np.sum(probs*mags)
        return mags_avgs

for t in T:
    path = f'temp_{t:.2f}'
    ising.simulate(True, t, N, 1000, path)
    mags_avgs = average_magnetization(f'resultados/mags_{path}.npy', f'resultados/probs_{path}.npy')
    np.save(f'resultados/mags_avgs_{path}.npy', mags_avgs)