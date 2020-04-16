import numpy as np, types, warnings, multiprocessing
from copy import deepcopy
from joblib import Parallel, delayed
import pandas as pd

class _LinUCBnTSSingle:
    def __init__(self, alpha, context_dim):
        self.alpha = alpha
        if 'Ainv' not in dir(self):
            self.Ainv = np.eye(context_dim)
            self.b = np.zeros((context_dim, 1))
    def _sherman_morrison_update(self, Ainv, x):
        Ainv -= np.linalg.multi_dot([Ainv, x, x.T, Ainv]) / (1.0 + np.linalg.multi_dot([x.T, Ainv, x]))

    def fit(self, X, y):
        if len(X.shape) == 1:
            X = X.reshape((1, -1))
        self.Ainv = np.eye(X.shape[1])
        self.b = np.zeros((X.shape[1], 1))

        self.partial_fit(X,y)

    def partial_fit(self, X, y):
        if len(X.shape) == 1:
            X = X.reshape((1, -1))
        if 'Ainv' not in dir(self):
            self.Ainv = np.eye(X.shape[1])
            self.b = np.zeros((X.shape[1], 1))
        sumb = np.zeros((X.shape[1], 1))
        for i in range(X.shape[0]):
            x = X[i, :].reshape((-1, 1))
            r = y[i]
            sumb += r * x
            self._sherman_morrison_update(self.Ainv, x)

        self.b += sumb

    def predict(self, X, exploit=False):
        if len(X.shape) == 1:
            X = X.reshape((1, -1))

        pred = self.Ainv.dot(self.b).T.dot(X.T).reshape(-1)

        if not exploit:
            return pred

        for i in range(X.shape[0]):
            x = X[i, :].reshape((-1, 1))
            cb = self.alpha * np.sqrt(np.linalg.multi_dot([x.T, self.Ainv, x]))
            pred[i] += cb[0]

        return pred

    def exploit(self, X):
        return self.predict(X, exploit = True)
