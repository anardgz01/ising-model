import numpy as np
import ising_simulator as ising
import os
import json
import plot_animation_ising as myplot

N = 16
T = np.random.uniform(1.5, 3.5, 10)
T = np.sort(T)

# Delete all contents in 'resultados' folder
folder_path = '/home/anardgz01/Documentos/GitHub/ising-model/resultados'
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    if os.path.isfile(file_path):
        os.unlink(file_path)

def average_magnetization(file : str) -> tuple[float, float]:
        mags = np.load(file)
        # mags_100 = mags[np.arange(99, len(mags), 100)]
        mags_100 = mags[np.arange(99, len(mags), 100)]
        
        average = np.mean(mags_100)
        error = np.std(mags_100)
        return average, error

mags_dict = dict()

for t in T:
    path = f'temp_{t:.2f}'
    ising.simulate(True, t, N, 10000, path)
    mags_dict[t] = average_magnetization(f'resultados/mags_{path}.npy')

# mags_dict_str_keys = {str(key).replace(' ', ''): value for key, value in mags_dict.items()}
with open('resultados/mags_dict.json', 'w') as f:
    json.dump(mags_dict, f, indent=4)

myplot.plot_mag()