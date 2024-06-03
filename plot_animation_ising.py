import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from matplotlib.animation import PillowWriter
import numpy as np
import glob

def update(i):
    im.set_array(confs[i])
    return [im]
    
def plot():
    fps = 10
    global confs, im
    confs = np.load('resultados/confs_.npy')

    fig, ax = plt.subplots()
    im = ax.imshow(confs[0], cmap=matplotlib.colors.ListedColormap(['purple', 'lavender']), vmin=-1, vmax=1)

    ani = anim.FuncAnimation(fig, update, frames=range(len(confs)), interval=((1/fps)*1000), blit=True)

    # ani.save("animation_N_64_T_2_5_s_1000.gif", writer=PillowWriter(fps=10))

    plt.show()

def plot_mag():
    exps = np.load('resultados/mags_exps.npy')

    fig, ax = plt.subplots()

    ax.plot(exps[0], exps[1])

    
    ax.set_xlabel('Temperaturas (KT)')
    ax.set_ylabel('Magnetización promedio')
    ax.set_title('Evolución de la magnetización promedio en función de la temperatura')
    ax.set_xticks(exps[0])
    ax.set_xticklabels([f'{temp:.2f}' for temp in exps[0]])

    plt.show()

def plot_voluntario():
    # Load data
    data_avgs = [np.load(f'resultados/matrices_voluntario/avgs_matrix_{name}.npy') for name in ['mags', 'energies', 'heats']]
    data_errs = [np.load(f'resultados/matrices_voluntario/stderr_matrix_{name}.npy') for name in ['mags', 'energies', 'heats']]

    # Create figures and axes
    figs, axes = zip(*[plt.subplots(ncols=2, figsize=(6, 6), constrained_layout=True) for _ in range(3)])

    N = np.array([16, 32, 64, 128])
    # T = np.array([1.50, 1.72, 1.94, 2.17, 2.39, 2.61, 2.83, 3.06, 3.28, 3.50])
    T = np.array([2.15, 2.19, 2.23, 2.27, 2.31, 2.34, 2.38, 2.42, 2.46, 2.50])

    labels = ['Average Magnetization', 'Average Energy', 'Specific Heat']
    # titles = ['Evolution of average magnetization as a function of temperature', 'Evolution of average energy as a function of temperature', 'Evolution of specific heat as a function of temperature']

    for avgs, errs, axs, label in zip(data_avgs, data_errs, axes, labels):
        for i in range(1, 5):
            # Normalization of the energy, if you comment this line, the energy will be plotted as it appears in the slides
            if label == 'Average Energy':
                avgs[1:, i] = avgs[1:, i]/(N[i-1])
                errs[1:, i] = errs[1:, i]/(N[i-1])
            axs[0].errorbar(avgs[1:, 0], avgs[1:, i], yerr=np.abs(errs[1:, i]), label=f'N = {N[i-1]}', marker='o', markersize=3, capsize=3)
        for i in range(1, 11):
            axs[1].errorbar(avgs[0, 1:], avgs[i, 1:], yerr=np.abs(errs[i, 1:]), label=f'T = {T[i-1]}', marker='o', markersize=3, capsize=3)

        axs[0].set_xlabel('Temperatures (KT)')
        axs[0].set_ylabel(label)
        axs[1].set_xlabel('N')
        axs[1].set_ylabel(label)

        axs[0].set_title(f'Evolution of {label.lower()} as a function of temperature')
        axs[1].set_title(f'Evolution of {label.lower()} as a function of N')

        axs[0].legend(loc=2, bbox_to_anchor=(1, 1))
        axs[1].legend(loc=2, bbox_to_anchor=(1, 1))

        axs[0].set_xticks(T)
        axs[1].set_xticks(N)

    corrfig, corrax = plt.subplots(2, 2, figsize=(10, 10))
    corrfig.suptitle('Correlation as a function of distance')

    files = glob.glob('resultados/correlations_global_N_*_temp_*.npy')

    def extract_nt(file):
        name_split = file.split('_')
        n = int(name_split[3])
        t = float(name_split[5].split('.npy')[0])
        return n, t

    files = sorted(glob.glob('resultados/correlations_global_N_*_temp_*.npy'), key=extract_nt)

    def convert_index(index):
        return index // 2, index % 2

    for file in files:
        name_split = file.split('_')
        n = float(name_split[3])
        t = name_split[5]
        t = float(t.split('.npy')[0]) 
        corrdata = np.load(file)
        # print(np.where(N == n)[0][0])
        corrax[convert_index(np.where(N == n)[0][0])].errorbar(np.arange(len(corrdata[0])), corrdata[0], label=f'T = {t:.2f}')
        # matrix[t_values_augmented == t, n_values_augmented == n] = data[0]

    for i in range(4):
        corrax[convert_index(i)].set_ylabel('Correlation')
        corrax[convert_index(i)].set_xlabel('Distance (i)')
        corrax[convert_index(i)].set_title(f'N = {N[i]}')
        corrax[convert_index(i)].set_xticks(np.arange(N[i]//2))
        corrax[convert_index(i)].set_xticklabels(np.arange(1, N[i]//2+1))
    corrax[1,1].set_xticks(np.arange(3,N[3]//2, step=4))
    corrax[1,0].set_xticks(np.arange(1,N[2]//2, step=2))
    handles, labels = corrax[convert_index(0)].get_legend_handles_labels()
    corrfig.legend(handles, labels, loc='center right')

    plt.show()

if __name__ == '__main__':
    plot_voluntario()