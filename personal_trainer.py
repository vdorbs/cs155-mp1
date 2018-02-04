import sys
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

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
        'bijan': [],
        'victor': [
        ],
        'kristjan': [
            Test(
                'MultinomialNB',
                MultinomialNB(),
                tf_idf_with_lengths(x), y
            ),
            Test(
                'GaussianNB',
                MultinomialNB(),
                tf_idf_with_lengths(x), y
            ),
            Test(
                'Gridsearch pipeline SGD',
                GridSearchCV(Pipeline([
                    ('tfidf', TfidfTransformer()),
                    ('clf', SGDClassifier(loss='hinge',
                                           penalty='l1',
                                           random_state=42,
                                           max_iter=10,
                                           tol=None))]),
                             {'clf__alpha': (1e-2, 1e-3)},
                            n_jobs=-1),
                x, y
            ),
            Test(
                'Gridsearch pipeline Bayes',
                GridSearchCV(Pipeline([
                    ('tfidf', TfidfTransformer()),
                    ('clf', MultinomialNB())]),
                             {'clf__alpha': (1e-2, 1e-3)},
                            n_jobs=-1),
                x, y
            ),
            Test(
                'RandomForestClassifier',
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=15,
                    n_jobs = -1
                ), tf_idf(x), y
            )
        ]
    }[player]

def personal_trainer(path, player):
    data = np.load(path)
    x, y = data[:,1:], data[:,0]

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
