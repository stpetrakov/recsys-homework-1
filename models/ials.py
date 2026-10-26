import numpy as np
from .base import Recommender


class IALS(Recommender):
    """Exact alternating normal equations; all missing pairs have p=0, c=1.

    Every stored review has confidence 1+alpha, including an explicit zero
    target. Do not eliminate zeros from the target CSR matrix.
    """

    def __init__(self, factors=32, regularization=1.0, alpha=20.0, iterations=10, seed=42):
        self.factors, self.regularization = factors, regularization
        self.alpha, self.iterations, self.seed = alpha, iterations, seed

    @staticmethod
    def update_rows(targets, fixed, alpha, regularization):
        base = fixed.T @ fixed + regularization * np.eye(fixed.shape[1])
        out = np.zeros((targets.shape[0], fixed.shape[1]))
        for row in range(targets.shape[0]):
            lo, hi = targets.indptr[row:row + 2]
            idx, p = targets.indices[lo:hi], targets.data[lo:hi]
            y = fixed[idx]
            a = base + alpha * (y.T @ y)
            b = y.T @ ((1 + alpha) * p)
            out[row] = np.linalg.solve(a, b)
        return out

    def fit(self, targets):
        targets = targets.astype(np.float64).tocsr()
        transposed = targets.T.tocsr()
        rng = np.random.default_rng(self.seed)
        self.items = rng.normal(0, 0.05, (targets.shape[1], self.factors))
        self.loss_history = []
        for _ in range(self.iterations):
            self.users = self.update_rows(targets, self.items, self.alpha, self.regularization)
            self.items = self.update_rows(transposed, self.users, self.alpha, self.regularization)
            self.loss_history.append(self.objective(targets))
        return self

    def objective(self, targets):
        # Sum over unobserved pairs without materializing the dense score matrix.
        loss = np.sum((self.users.T @ self.users) * (self.items.T @ self.items))
        coo = targets.tocoo()
        pred = np.einsum("ij,ij->i", self.users[coo.row], self.items[coo.col])
        loss += np.sum((1 + self.alpha) * (coo.data - pred) ** 2 - pred ** 2)
        return float(loss + self.regularization *
                     (np.sum(self.users ** 2) + np.sum(self.items ** 2)))

    def score(self, users):
        return self.users[users] @ self.items.T

