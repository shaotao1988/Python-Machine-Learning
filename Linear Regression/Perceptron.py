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

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
        self.w_ = []
        self.errors = []

    def fit(self, X, y):
        """Fit training data
        :param X: Training data, shape = [n_samples, n_features]
        :param y: Target values, shape = [n_samples]
        :return: self
        """
        self.w_ = np.zeros((X.shape[1] + 1, 1))
        self.errors = np.zeros((self.n_iter, 1))
        X = np.insert(X, 0, 1, 1) #Insert x0
        for i in range(self.n_iter):
            output = self.net_input(X)
            error = output - y
            update = self.eta * error
            self.w_ -= np.dot(X.T, update)
            self.errors[i] = np.dot(error.T, error) / 2
        return self

    def net_input(self, X):
        return np.dot(X, self.w_)

    def predict(self, X):
        X = np.insert(X, 0, 1, 1)  # Insert x0
        return np.where(self.net_input(X) > 0, 1, -1)

    def normalize(self, X):
        X = (X-X.mean(0))/X.std(0)
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
    
    #plot class samples
    for idx, cl in enumerate(np.unique(y)):
        filter = (y.ravel() == cl)
        plt.scatter(x=X[filter, 0], y = X[filter, 1],
                    alpha = 0.4, c = 'green',
                    marker = 'o', label = cl)

if __name__ == "__main__":
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1).reshape(y.shape[0], 1)
    X = df.iloc[0:100, [0, 2]].values
    Classifier = Perceptron(0.001, 100)
    X_std = Classifier.normalize(X)
    plt.scatter(X_std[:50, 0], X_std[:50, 1], color='red', marker='s', label='setosa')
    plt.scatter(X_std[50:100, 0], X_std[50:100, 1], color='blue', marker='x', label='versicolor')
    plt.legend(loc='upper left')
    Classifier.fit(X_std, y)
    Z = Classifier.predict(X_std)
    plot_decision_regions(X_std, y, classifier=Classifier)
    plt.show()
    plt.plot(Classifier.errors)
    plt.show()
