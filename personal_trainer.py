import sys
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier

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

def tf_idf(x):
    idf = np.log(x.shape[0] / np.apply_along_axis(np.count_nonzero, 0, x))
    tf = (x.T / np.sum(x.T + np.finfo(float).eps, 0)).T
    return tf * idf

def tests_by_player(player, x, y):
    return {
        'bijan': [
            Test(
                'KNN unnormalized', 
                KNeighborsClassifier(algorithm = 'kd_tree', n_neighbors = 10, weights = 'uniform'), 
                x, y
            )
        ],
        'victor': [
            Test(
                'Unnormalized',
                SGDClassifier(max_iter=10),
                x, y
            ),
            Test(
                'Unnormalized with lengths',
                SGDClassifier(max_iter=10),
                unnormalized_with_lengths(x), y
            ),
            Test(
                'tf-idf',
                SGDClassifier(max_iter=10),
                tf_idf(x), y
            ),
            Test(
                'tf-idf with lengths',
                SGDClassifier(max_iter=10),
                tf_idf_with_lengths(x), y
            )
        ],
        'kristjan': []
    }[player]

def personal_trainer(path, player):
    data = np.load(path)
    x, y = data[:,1:], data[:,1]
    
    tests = tests_by_player(player, x, y)
    for test in tests:
        test.fit()
        print(test.name, test.score())

def unnormalized_with_lengths(x):
    return np.append(x, np.reshape(np.sum(x, 1), (-1, 1)), 1)

def tf_idf_with_lengths(x):
    x_sums = np.reshape(np.sum(x, 1), (-1, 1))
    x_normal = tf_idf(x)
    return np.append(x_normal, x_sums, 1)

if __name__ == '__main__':
    try:
        personal_trainer(sys.argv[1], sys.argv[2])
    except IndexError as e:
        print('FEED ME DATA')
        print('   (ಠ‿ಠ)')
