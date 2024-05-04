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

if __name__ == '__main__':
    plot_mag()