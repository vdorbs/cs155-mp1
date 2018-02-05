import sys
import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import AdaBoostClassifier
import time

class Test:
    def __init__(self, name, model, x, y, noise=False):
        self.name = name
        self.model = model
        if noise:
            self.x = add_noise(x)
        else:
            self.x = x
        self.y = y
        self.noise = noise

    def fit(self):
        self.model.fit(self.x, self.y)

    def score(self):
        return self.model.score(self.x, self.y)

class KFoldTest:
    def __init__(self, test, n_splits=5, noise=False):
        self.name = test.name
        self.models = [clone(test.model) for i in range(n_splits)]
        self.kf = KFold(n_splits=n_splits)
        self.x = test.x
        self.y = test.y
        self.noise = noise

    def fit(self):
        for model, (train, _) in zip(self.models, self.kf.split(self.x)):
            if self.noise:
                x = add_noise(self.x[train])
            else:
                x = self.x[train]
            y = self.y[train]
            model.fit(x, y)

    def score(self):
        scores = []
        for model, (_, test) in zip(self.models, self.kf.split(self.x)):
            x, y = self.x[test], self.y[test]
            scores.append(model.score(x, y))
        scores = np.array(scores)
        return np.mean(scores), np.std(scores)

def tf_idf(x, idf=None):
    if idf is None:
        idf = np.log(x.shape[0] / np.apply_along_axis(np.count_nonzero, 0, x))
    tf = (x.T / np.sum(x.T + np.finfo(float).eps, 0)).T
    return tf * idf, idf

def add_noise(x, scale=0.0005):
    # noise = np.random.normal(scale=scale, size=x.shape)
    # return x + noise
    nx = np.copy(x)
    nx[np.where(nx != 0)] += np.random.normal(scale=scale, size=np.count_nonzero(x))
    return nx

def tests_by_player(player, x, y):
    x_tf_idf, _ = tf_idf(x)
    return {
        'bijan': [
         ],
        'victor': [
        ],
        'kristjan': [
            Test(
                'LinearSVC w/ noise',
                LinearSVC(),
                x, y,
                noise=True
            ),
            KFoldTest(
                Test(
                    'LinearSVC w/ noise',
                    LinearSVC(),
                    x, y,
                    noise=True
                )
            ),
            KFoldTest(
                Test(
                    'LinearSVC w/o noise',
                    LinearSVC(),
                    x, y,
                    noise=False
                )
            )
        ]
    }[player]

def personal_trainer(path, player):
    data = np.load(path)
    x, y = data[:,1:], data[:,0]
    x, idf = tf_idf(x)
    tests = tests_by_player(player, x, y)
    for test in tests:
        test.fit()
        print(test.name, test.score())

    if type(tests[0]).__name__ == 'Test':
        return tests[0].model, idf
    else:
        return tests[0].models[0], idf

def personal_prophet(path, model, idf):
    x = np.load(path)
    x, _ = tf_idf(x, idf)
    return model.predict(x)

def tf_idf_with_lengths(x):
    x_sums = np.reshape(np.sum(x, 1), (-1, 1))
    x_normal = tf_idf(x)
    return np.append(x_normal, x_sums, 1)

def normalize(x):
    return (x - np.mean(x, 0)) / np.std(x, 0)

if __name__ == '__main__':
    try:
        model, idf = personal_trainer(sys.argv[1], sys.argv[2])
        if '--test' in sys.argv:
            pred = personal_prophet(sys.argv[4], model, idf)
            lines = ['{},{}'.format(i + 1, int(p)) for i, p in enumerate(pred)]
            timestr = time.strftime("%Y%m%d-%H%M%S")
            with open('predictions_{}.csv'.format(timestr), 'w') as fh:
                fh.write('Id,Prediction\n')
                fh.write('\n'.join(lines))
    except IndexError as e:
        print('FEED ME DATA')
        print('   (ಠ‿ಠ)')
