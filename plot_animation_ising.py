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
    mags_avgs = np.load('resultados/matrices_voluntario/avgs_matrix_mags.npy')
    energies_avgs = np.load('resultados/matrices_voluntario/avgs_matrix_energies.npy')
    heat_avgs = np.load('resultados/matrices_voluntario/avgs_matrix_heats.npy')
    correlations_avgs = np.load('resultados/matrices_voluntario/avgs_matrix_correlations.npy')
    mags_errs = np.load('resultados/matrices_voluntario/stderr_matrix_mags.npy')
    energies_errs = np.load('resultados/matrices_voluntario/stderr_matrix_energies.npy')
    heat_errs = np.load('resultados/matrices_voluntario/stderr_matrix_heats.npy')
    correlations_errs = np.load('resultados/matrices_voluntario/stderr_matrix_correlations.npy')

    fig_mags, axes_mags = plt.subplots(2)
    fig_energies, axes_energies = plt.subplots(2)
    fig_heat, axes_heat = plt.subplots(2)
    fig_correlations, axes_correlations = plt.subplots(2)

    N = np.array([16, 32, 64, 128])
    T = np.array([1.50, 1.72, 1.94, 2.17, 2.39, 2.61, 2.83, 3.06, 3.28, 3.50])

    for i in range(1,5):
        axes_mags[0].errorbar(mags_avgs[1:, 0], mags_avgs[1:, i], yerr=mags_errs[1:, i], label=f'N = {N[i-1]}', marker='o', markersize=3, capsize=3)
        axes_energies[0].errorbar(energies_avgs[1:, 0], energies_avgs[1:,i], yerr=energies_errs[1:, i], label=f'N = {N[i-1]}', marker='o', markersize=3, capsize=3)
        axes_heat[0].errorbar(heat_avgs[1:, 0], heat_avgs[1:, i], yerr=heat_errs[1:,i ], label=f'N = {N[i-1]}', marker='o', markersize=3, capsize=3)
        axes_correlations[0].errorbar(correlations_avgs[1:, 0], correlations_avgs[1:, i], yerr=correlations_errs[1:, i], label=f'N = {N[i-1]}', marker='o', markersize=3, capsize=3)

    for i in range(1,11):
        axes_mags[1].errorbar(mags_avgs[0, 1:], mags_avgs[i, 1:], yerr=mags_errs[i, 1:], label=f'T = {T[i-1]}', marker='o', markersize=3, capsize=3)
        axes_energies[1].errorbar(energies_avgs[0, 1:], energies_avgs[i, 1:], yerr=energies_errs[i, 1:], label=f'T = {T[i-1]}', marker='o', markersize=3, capsize=3)
        axes_heat[1].errorbar(heat_avgs[0, 1:], heat_avgs[i, 1:], yerr=heat_errs[i, 1:], label=f'T = {T[i-1]}', marker='o', markersize=3, capsize=3)
        axes_correlations[1].errorbar(correlations_avgs[0, 1:], correlations_avgs[i, 1:], yerr=correlations_errs[i, 1:], label=f'T = {T[i-1]}', marker='o', markersize=3, capsize=3)
    
    axes_mags[0].set_xlabel('Temperaturas (KT)')
    axes_mags[0].set_ylabel('Magnetización promedio')
    axes_mags[1].set_xlabel('N')
    axes_mags[1].set_ylabel('Magnetización promedio')

    axes_energies[0].set_xlabel('Temperaturas (KT)')
    axes_energies[0].set_ylabel('Energía promedio')
    axes_energies[1].set_xlabel('N')
    axes_energies[1].set_ylabel('Energía promedio')

    axes_heat[0].set_xlabel('Temperaturas (KT)')
    axes_heat[0].set_ylabel('Calor específico')
    axes_heat[1].set_xlabel('N')
    axes_heat[1].set_ylabel('Calor específico')

    axes_correlations[0].set_xlabel('Temperaturas (KT)')
    axes_correlations[0].set_ylabel('Correlación promedio')
    axes_correlations[1].set_xlabel('N')
    axes_correlations[1].set_ylabel('Correlación promedio')

    axes_mags[0].set_title('Evolución de la magnetización promedio en función de la temperatura')
    axes_mags[1].set_title('Evolución de la magnetización promedio en función de N')

    axes_energies[0].set_title('Evolución de la energía promedio en función de la temperatura')
    axes_energies[1].set_title('Evolución de la energía promedio en función de N')

    axes_heat[0].set_title('Evolución del calor específico en función de la temperatura')
    axes_heat[1].set_title('Evolución del calor específico en función de N')

    axes_correlations[0].set_title('Evolución de la correlación promedio en función de la temperatura')
    axes_correlations[1].set_title('Evolución de la correlación promedio en función de N')

    axes_mags[0].legend(loc=2, bbox_to_anchor=(1, 1))
    axes_energies[0].legend(loc=2, bbox_to_anchor=(1, 1))
    axes_heat[0].legend(loc=2, bbox_to_anchor=(1, 1))
    axes_correlations[0].legend(loc=2, bbox_to_anchor=(1, 1))
    axes_mags[1].legend(loc=2, bbox_to_anchor=(1, 1))
    axes_energies[1].legend(loc=2, bbox_to_anchor=(1, 1))
    axes_heat[1].legend(loc=2, bbox_to_anchor=(1, 1))
    axes_correlations[1].legend(loc=2, bbox_to_anchor=(1, 1))

    axes_mags[0].set_xticks(mags_avgs[1:, 0])
    axes_energies[0].set_xticks(energies_avgs[1:, 0])
    axes_heat[0].set_xticks(heat_avgs[1:, 0])
    axes_correlations[0].set_xticks(correlations_avgs[1:, 0])

    axes_mags[1].set_xticks(N)
    axes_energies[1].set_xticks(N)
    axes_heat[1].set_xticks(N)
    axes_correlations[1].set_xticks(N)

    plt.show()

if __name__ == '__main__':
    plot_voluntario()