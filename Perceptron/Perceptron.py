import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class Perceptron(object):
    """Perceptron classifier.

    Parameters
    -----------
    eta: float
        Learning rate(between 0 and 1.0)
    n_iter: int
        Loop time for learning process

    Attributes
    -----------
    w_: ld-array
        Weights after learning
    errors_: list
        Number of misclassifications in each epoch
    """

    def __init__(self, eta = 0.01, n_iter = 10):
        self.eta = eta
        self.n_iter = n_iter
        self.w_ = []
        self.errors = []

    def fit(self, X, y):
        """Fit training data
        :param X: Training data, shape = [n_samples, n_features]
        :param y: Target values, shape = [n_examples]
        :return: self
        """
        self.w_ = np.zeros(X.shape[1] + 1)
        self.errors = np.zeros(self.n_iter)
        for i in range(self.n_iter):
            update = self.eta*(self.predict(X) - y)
            self.w_[1:] -= np.dot(X.T, update)
            self.w_[0] -= sum(update)
            self.errors[i] = np.dot(update, update)/X.shape[0]
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        positive = np.ones(X.shape[0])
        negative = -1*positive
        return np.where(self.net_input(X) > 0, positive, negative)

def plot_decision_regions(X, y, classifier, resolution = 0.02):
    markers = ('s', 'o', 'x', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    #plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha = 0.4, cmap = cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x = X[y == cl, 0], y = X[y == cl, 1],
                    alpha = 0.8, c=cmap(idx),
                    marker = markers[idx], label = cl)


if __name__ == "__main__":
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header = None)
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)
    X = df.iloc[0:100, [0, 2]].values
    plt.scatter(X[:50, 0], X[:50, 1], color = 'red', marker = 'o', label = 'setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
    plt.legend(loc = 'upper left')
    plt.show()
    Classifier = Perceptron()
    Classifier.fit(X, y)
    #plt.plot(Classifier.errors)
    plot_decision_regions(X, y, classifier = Classifier)
    plt.show()
    print('finished')






















