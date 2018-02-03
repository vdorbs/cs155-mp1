import numpy as np
import sys

def prep_data(path):
    data = np.loadtxt(path, skiprows=1)
    np.save('{}.npy'.format(path.rsplit('.', 1)[0]), data)

if __name__ == '__main__':
    try:
        prep_data(sys.argv[1])
    except IndexError as e:
        print('Oi, pass me a file name you idiot.')
