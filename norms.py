import numpy as np
from sklearn.metrics import euclidean_distances
from sklearn.base import BaseEstimator, TransformerMixin
from convert import convert

class OrigN(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.func = 0

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_new = X['data'].copy()
        return X_new
class OrigS(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.func = 0

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_new = X
        return X_new
class SpectralNorm(BaseEstimator, TransformerMixin):
    def __init__(self, func=0):
        self.func = func

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_new = X.copy()
        [np.fill_diagonal(X_new[i], 0) for i in range(X_new.shape[0])]
        degrees = np.array([np.diag(1 / np.sqrt(np.nansum(X_new[i], axis=1))) for i in range(X_new.shape[0])])
        normed_X = np.array([degrees[i].dot(X_new[i]).dot(degrees[i]) for i in range(X_new.shape[0])])
        return normed_X
class BinarNorm(BaseEstimator, TransformerMixin):
    def __init__(self, func=0):
        self.func = func

    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        bin_X = X['data'].copy()
        bin_X[bin_X > 0] = 1
        return bin_X

class WbysqDist(BaseEstimator, TransformerMixin):
    def __init__(self, func=0):
        self.func = func
    def fit(self, X, y=None):
        return self
    def distance(self, d):
        if len(d.shape) == 2:
            dist = euclidean_distances(d)
            np.fill_diagonal(dist, 1)
        else:
            dist = np.array([euclidean_distances(d[i]) for i in range(d.shape[0])])
            [np.fill_diagonal(dist[i],1) for i in range(d.shape[0])]
        return dist

    def transform(self, X, y=None):
        
        dist = self.distance(X['dist'])
        
        if len(dist.shape) == 2:

            weighted_X = np.array([X['data'][i] / dist ** 2 for i in range(X['data'].shape[0])])
        else:

            weighted_X = np.array([X['data'][i] / dist[i] ** 2 for i in range(X['data'].shape[0])])
            
        [np.fill_diagonal(weighted_X[i], 0) for i in range(X['data'].shape[0])]
        return weighted_X
