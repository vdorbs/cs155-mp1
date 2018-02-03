import sys
import numpy as np

def tf_idf(x):
    idf = np.log(x.shape[0] / np.apply_along_axis(np.count_nonzero, 0, x))
    tf = (x.T/x.sum()).T
    return tf * idf

def personal_trainer(path):
    data = np.load(path)
    print(data.shape)
    x, y = data[:,1:], data[:,1]
    print(x.shape)
    print(y.shape)
    tf_idf(x)

if __name__ == '__main__':
    try:
        personal_trainer(sys.argv[1])
    except IndexError as e:
        print('FEED ME DATA')
        print('   (ಠ‿ಠ)')
