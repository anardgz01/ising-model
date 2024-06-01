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
    data_avgs = [np.load(f'resultados/matrices_voluntario/avgs_matrix_{name}.npy') for name in ['mags', 'energies', 'heats', 'correlations']]
    data_errs = [np.load(f'resultados/matrices_voluntario/stderr_matrix_{name}.npy') for name in ['mags', 'energies', 'heats', 'correlations']]

    # Create figures and axes
    figs, axes = zip(*[plt.subplots(2) for _ in range(4)])

    N = np.array([16, 32, 64, 128])
    T = np.array([1.50, 1.72, 1.94, 2.17, 2.39, 2.61, 2.83, 3.06, 3.28, 3.50])

    labels = ['Magnetización promedio', 'Energía promedio', 'Calor específico', 'Correlación promedio']
    titles = ['Evolución de la magnetización promedio en función de la temperatura', 'Evolución de la energía promedio en función de la temperatura', 'Evolución del calor específico en función de la temperatura', 'Evolución de la correlación promedio en función de la temperatura']

    for avgs, errs, (fig, ax), label, title in zip(data_avgs, data_errs, axes, labels, titles):
        for i in range(1, 5):
            ax[0].errorbar(avgs[1:, 0], avgs[1:, i], yerr=errs[1:, i], label=f'N = {N[i-1]}', marker='o', markersize=3, capsize=3)
        for i in range(1, 11):
            ax[1].errorbar(avgs[0, 1:], avgs[i, 1:], yerr=errs[i, 1:], label=f'T = {T[i-1]}', marker='o', markersize=3, capsize=3)

        ax[0].set_xlabel('Temperaturas (KT)')
        ax[0].set_ylabel(label)
        ax[1].set_xlabel('N')
        ax[1].set_ylabel(label)

        ax[0].set_title(title)
        ax[1].set_title(f'Evolución de {label.lower()} en función de N')

        ax[0].legend(loc=2, bbox_to_anchor=(1, 1))
        ax[1].legend(loc=2, bbox_to_anchor=(1, 1))

        ax[0].set_xticks(avgs[1:, 0])
        ax[1].set_xticks(N)

    plt.show()

if __name__ == '__main__':
    plot_voluntario()