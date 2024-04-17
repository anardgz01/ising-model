import numpy as np
import ising_simulator as ising
import os
import plot_animation_ising as myplot

N = 16
T = np.random.uniform(1.5, 3.5, 10)
T = np.sort(T)

# Delete all contents in 'resultados' folder
folder_path = '/home/anardgz01/Documentos/GitHub/ising-model/resultados'
folder_path = 'resultados'
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    if os.path.isfile(file_path):
        os.unlink(file_path)

def average_magnetization(file : str) -> tuple[float, float]:
        mags = np.load(file)
        mags = np.delete(mags, -1)
        mags_2d = mags.reshape((100, -1))
        mags_avgs = np.mean(mags_2d, axis=0)
        mags_stds = np.std(mags_2d, axis=0)
        return mags_avgs, mags_stds

for t in T:
    path = f'temp_{t:.2f}'
    ising.simulate(True, t, N, 1000000, path)
for index, t in enumerate(T):
    path = f'temp_{t:.6f}'
    mags_avgs, mags_stds = average_magnetization(f'resultados/mags_{path}.npy')
    np.save(f'resultados/mags_avgs_{path}.npy', mags_avgs)
    np.save(f'resultados/mags_stds_{path}.npy', mags_stds)
    print(f'finished simulation {index+1} of {len(T)}')

myplot.plot_mag()
