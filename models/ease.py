import numpy as np
from scipy.linalg import cho_factor, cho_solve
from .base import Recommender


class EASE(Recommender):
    def __init__(self, regularization=100.0):
        self.regularization = regularization

    def fit(self, positive):
        if self.regularization <= 0:
            raise ValueError("Positive regularization required")
        self.x = positive.astype(np.float64).tocsr()
        gram = (self.x.T @ self.x).toarray()
        gram.flat[::gram.shape[0] + 1] += self.regularization
        precision = cho_solve(cho_factor(gram), np.eye(gram.shape[0]))
        self.weights = -precision / np.diag(precision)[None, :]
        np.fill_diagonal(self.weights, 0)
        return self

    def score(self, users):
        return self.x[users] @ self.weights

