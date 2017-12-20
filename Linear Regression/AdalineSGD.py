import numpy as np
import pandas as pd
from numpy.random import seed
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class AdalineSGD(object):
    """Adaptive Linear Neuron classifier with stochestic gradient descent.

    Parameters
    -----------
    eta: float
        Learning rate(between 0 and 1.0)
    n_iter: int
        Loop time for learning process
    shuffle: bool (default: True)
        Shuffles traing data in every epoch to prevent cycles
    random_state: int (default: None)
        set random state for shuffling and initializing the weights.

    Attributes
    -----------
    w_: ld-array
        Weights after learning
    errors: list
        Number of misclassifications in each epoch
    shuffle: bool (default: True)
        Shuffles traing data in every epoch to prevent cycles
    random_state: int (default: None)
        set random state for shuffling and initializing the weights.
    """

    def __init__(self, eta=0.01, n_iter=10,
                 shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.w_ = []
        self.errors = []
        if random_state:
            seed(random_state)

    def fit(self, X, y):
        """Fit training data
        :param X: Training data, shape = [n_samples, n_features]
        :param y: Target values, shape = [n_samples]
        :return: self
        """
        if not self.w_initialized:
            self._initialize_weights(X.shape[1] + 1)
        self.errors = []
        X = np.insert(X, 0, 1, 1)  # Insert x0
        for i in range(self.n_iter):
            error = 0
            if self.shuffle:
                X, y = self._shuffle(X, y)
            for xi, target in zip(X, y):
                error += self._update_weights(xi, target)
            self.errors.append(error)
        return self

    def partial_fit(self, X, y):
        """Training the data without reinitializing w"""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1] + 1)
        X = np.insert(X, 0, 1, 1)  # Insert x0
        if y.ravel().shape[0] > 1:  # may input multiple training data in one call
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        perms = np.random.permutation(X.shape[0])
        return X[perms], y[perms]

    def _initialize_weights(self, m):
        self.w_ = np.zeros(m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        output = self.net_input(xi)
        error = output - target
        update = self.eta * error
        self.w_ -= update * xi
        return 0.5 * error ** 2

    def net_input(self, X):
        return np.dot(X, self.w_)

    def predict(self, X):
        X = np.insert(X, 0, 1, 1)  # Insert x0
        return np.where(self.net_input(X) > 0, 1, -1)

    def normalize(self, X):
        X = (X - X.mean(0)) / X.std(0)
        return X


def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('s', 'o', 'x', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        f = (y.ravel() == cl)
        plt.scatter(x=X[f, 0], y=X[f, 1],
                    alpha=0.4, c='green',
                    marker='o', label=cl)


if __name__ == "__main__":
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1).reshape(y.shape[0], 1)
    X = df.iloc[0:100, [0, 2]].values
    Classifier = AdalineSGD(0.01, 10, random_state=1)
    X_std = Classifier.normalize(X)
    plt.scatter(X_std[:50, 0], X_std[:50, 1], color='red', marker='s', label='setosa')
    plt.scatter(X_std[50:100, 0], X_std[50:100, 1], color='blue', marker='x', label='versicolor')
    plt.legend(loc='upper left')
    # Classifier.fit(X_std, y)
    for _ in range(10):
        Classifier.partial_fit(X_std, y)
    Z = Classifier.predict(X_std)
    plot_decision_regions(X_std, y, classifier=Classifier)
    plt.show()
    plt.plot(range(1, len(Classifier.errors)+1), Classifier.errors, marker = 'o')
    plt.show()
