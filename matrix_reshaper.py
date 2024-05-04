import numpy as np
import glob

def reshape():
    mags = np.load('resultados/mags_exps.npy')
    heats = np.load('resultados/specific_heat_avgs.npy')
    energies = np.load('resultados/energies_avgs.npy')

    matrices = [mags, energies, heats]
    names = ['mags', 'energies', 'heats', 'correlations']
    n_values = np.unique(mags[0])
    t_values = np.unique(mags[1])

    n_values_augmented = np.insert(n_values, 0, 0)
    t_values_augmented = np.insert(np.around(t_values, 2), 0, 0)

    avgs_matrices = [np.zeros((len(t_values)+1, len(n_values)+1)) for _ in range(4)]
    stderr_matrices = [np.zeros((len(t_values)+1, len(n_values)+1)) for _ in range(4)]

    files = glob.glob('resultados/correlations_global_N_*_temp_*.txt')
    for index, matrix in enumerate(avgs_matrices):
        matrix[1:, 0] = t_values
        matrix[0, 1:] = n_values
            
        if index < 3:
            matrix[1:, 1:] = matrices[index][2].reshape(len(t_values), len(n_values))

        elif index == 3:    # Correlations
            for file in files:
                name_split = file.split('_')
                n = float(name_split[3])
                t = name_split[5]
                t = float(t.split('.txt')[0]) 
                data = np.loadtxt(file)
                matrix[t_values_augmented == t, n_values_augmented == n] = data[0]
            
        np.save(f'resultados/matrices_voluntario/avgs_matrix_{names[index]}.npy', matrix)
        np.savetxt(f'resultados/matrices_voluntario/avgs_matrix_{names[index]}.txt', matrix)


    for index, matrix in enumerate(stderr_matrices):
        matrix[1:, 0] = t_values
        matrix[0, 1:] = n_values
        
        if index < 3:
            matrix[1:, 1:] = matrices[index][3].reshape(len(t_values), len(n_values))

        elif index == 3:    # Correlations
            for file in files:
                name_split = file.split('_')
                n = float(name_split[3])
                t = name_split[5]
                t = float(t.split('.txt')[0]) 
                data = np.loadtxt(file)
                matrix[t_values_augmented == t, n_values_augmented == n] = data[1]

        np.save(f'resultados/matrices_voluntario/stderr_matrix_{names[index]}.npy', matrix)
        np.savetxt(f'resultados/matrices_voluntario/stderr_matrix_{names[index]}.txt', matrix)




    # reshaped_matrices = []
    # for matrix in matrices:
    #     # Sort the matrix by n and t
    #     sorted_indices = np.lexsort((matrix[1], matrix[0]))
    #     sorted_matrix = matrix[:, sorted_indices]

    #     # Reshape the matrix
    #     reshaped_matrix = sorted_matrix[2:].reshape(len(n_values), len(t_values), 2)
    #     reshaped_matrices.append(reshaped_matrix)

if __name__ == '__main__':
    reshape()
