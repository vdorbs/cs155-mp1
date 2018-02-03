import sys
import numpy as np
from sklearn.linear_model import SGDClassifier

class Test:
    def __init__(self, name, model):
        self.name = name
        self.model = model

    def fit(self, x, y):
        self.model.fit(x ,y)

    def score(self):
        return self.model.score



def tf_idf(x):
    idf = np.log(x.shape[0] / np.apply_along_axis(np.count_nonzero, 0, x))
    tf = (x.T / np.sum(x.T + np.finfo(float).eps, 0)).T
    tf_idf = tf * idf
    print(np.array_equiv(np.where(tf_idf != 0), np.where(x != 0)))

def personal_trainer(path):
    data = np.load(path)
    print(data.shape)
    x, y = data[:,1:], data[:,1]
    x_normal = tf_idf(x)
    # model = SGDClassifier()
    # model.fit(x, y)
    # y_hat = model.predict(x)
    # print(np.linalg.norm(y - y_hat, 1) / len(y))

if __name__ == '__main__':
    try:
        personal_trainer(sys.argv[1])
    except IndexError as e:
        print('FEED ME DATA')
        print('   (ಠ‿ಠ)')
