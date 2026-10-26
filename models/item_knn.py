import numpy as np
from scipy import sparse
from .base import Recommender


class ItemKNN(Recommender):
    def __init__(self, neighbors=50, shrinkage=10.0):
        self.neighbors, self.shrinkage = neighbors, shrinkage

    def fit(self, positive):
        self.x = positive.astype(np.float64).tocsr()
        co = (self.x.T @ self.x).toarray()
        norms = np.sqrt(np.diag(co))
        sim = co / np.maximum(norms[:, None] * norms[None, :], 1e-12)
        sim *= co / np.maximum(co + self.shrinkage, 1e-12)
        np.fill_diagonal(sim, 0)
        # Each target column keeps its K most similar source items.
        k = min(self.neighbors, sim.shape[0] - 1)
        idx = np.argsort(-sim, axis=0, kind="stable")[:k]
        cols = np.broadcast_to(np.arange(sim.shape[1]), idx.shape)
        self.similarity = sparse.csr_matrix((sim[idx, cols].ravel(),
                                             (idx.ravel(), cols.ravel())), shape=sim.shape)
        self.similarity.eliminate_zeros()
        return self

    def score(self, users):
        return (self.x[users] @ self.similarity).toarray()

