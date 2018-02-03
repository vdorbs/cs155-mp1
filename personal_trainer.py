import sys
import numpy as np
from sklearn.linear_model import SGDClassifier

def tf_idf(x):
    idf = np.log(x.shape[0] / np.apply_along_axis(np.count_nonzero, 0, x))
    tf = (x.T/x.sum()).T
    return tf * idf

def personal_trainer(path):
    data = np.load(path)
    print(data.shape)
    x, y = data[:,1:], data[:,1]
    x_normal = tf_idf(x)
    model = SGDClassifier()
    model.fit(x, y)
    y_hat = model.predict(x)
    print(np.linalg.norm(y - y_hat, 1) / len(y))

if __name__ == '__main__':
    try:
        personal_trainer(sys.argv[1])
    except IndexError as e:
        print('FEED ME DATA')
        print('   (ಠ‿ಠ)')
