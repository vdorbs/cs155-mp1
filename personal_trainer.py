import sys
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.base import clone
from sklearn.model_selection import KFold

class Test:
    def __init__(self, name, model, x, y):
        self.name = name
        self.model = model
        self.x = x
        self.y = y

    def fit(self):
        self.model.fit(self.x, self.y)

    def score(self):
        return self.model.score(self.x, self.y)

class KFoldTest:
    def __init__(self, test, n_splits=5):
        self.name = test.name
        self.models = [clone(test.model) for i in range(n_splits)]
        self.kf = KFold(n_splits=n_splits)
        self.x = test.x
        self.y = test.y

    def fit(self):
        for model, (train, _) in zip(self.models, self.kf.split(self.x)):
            x, y = self.x[train], self.y[train]
            model.fit(x, y)

    def score(self):
        scores = []
        for model, (_, test) in zip(self.models, self.kf.split(self.x)):
            x, y = self.x[test], self.y[test]
            scores.append(model.score(x, y))
        scores = np.array(scores)
        return np.mean(scores), np.std(scores)

def tf_idf(x):
    idf = np.log(x.shape[0] / np.apply_along_axis(np.count_nonzero, 0, x))
    tf = (x.T / np.sum(x.T + np.finfo(float).eps, 0)).T
    return tf * idf

def tests_by_player(player, x, y):
    return {
        'bijan': [],
        'victor': [
            KFoldTest(
                Test(
                    'standard',
                    LinearSVC(),
                    x, y
                )
            ),
            KFoldTest(
                Test(
                    'tf-idf',
                    LinearSVC(),
                    tf_idf(x), y
                )
            )
        ],
        'kristjan': []
    }[player]

def personal_trainer(path, player):
    data = np.load(path)
    x, y = data[:,1:], data[:,0]

    tests = tests_by_player(player, x, y)
    for test in tests:
        test.fit()
        print(test.name, test.score())

def tf_idf_with_lengths(x):
    x_sums = np.reshape(np.sum(x, 1), (-1, 1))
    x_normal = tf_idf(x)
    return np.append(x_normal, x_sums, 1)

def normalize(x):
    return (x - np.mean(x, 0)) / np.std(x, 0)

if __name__ == '__main__':
    try:
        personal_trainer(sys.argv[1], sys.argv[2])
    except IndexError as e:
        print('FEED ME DATA')
        print('   (ಠ‿ಠ)')
