import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as anim
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

    plt.show()

def plot_mag():
    files_avgs = glob.glob('resultados/mags_avgs*.npy')
    files_stds = glob.glob('resultados/mags_stds*.npy')
    files_avgs.sort()
    files_stds.sort()
    mags_avgs_data = [np.load(file) for file in files_avgs]
    # mags_stds_data = [np.load(file) for file in files_stds]

    fig, ax = plt.subplots()

    for i in range(len(mags_avgs_data)):
        name = files_avgs[i]
        name = name.replace('resultados/mags_avgs_temp_', '')
        name = name.replace('.npy', '')

        ax.plot(range(len(mags_avgs_data[i])), mags_avgs_data[i], label = f'{name} K')
        # ax.errorbar(range(len(mags_avgs_data[i])), mags_avgs_data[i], yerr=mags_stds_data[i], marker='o', elinewidth=1, barsabove=True, label = f'{name} K')
        # ax.errorbar(range(len(mags_avgs_data[i])), mags_avgs_data[i], yerr=mags_stds_data[i], label = f'{name} K')

    
    ax.set_xlabel('Pasos Monte Carlo')
    ax.set_ylabel('Magnetización')
    ax.set_title('Evolución de la magnetización para distintas temperaturas')
    ax.legend()

    plt.show()

if __name__ == '__main__':
    plot_mag()