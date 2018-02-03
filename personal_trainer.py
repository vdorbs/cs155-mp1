import sys
import numpy as np

def personal_trainer(path):
    data = np.load(path)
    print(data.shape)
    x, y = data[:,1:], data[:,1]
    print(x.shape)
    print(y.shape)

if __name__ == '__main__':
    try:
        personal_trainer(sys.argv[1])
    except IndexError as e:
        print('FEED ME DATA')
        print('   (ಠ‿ಠ)')
