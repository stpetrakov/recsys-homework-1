import numpy as np


class Recommender:
    """Shared ranking policy: all items, exclude every previously reviewed item."""

    def recommend(self, users, seen, k=10):
        users = np.asarray(users, dtype=int)
        scores = np.asarray(self.score(users), dtype=float)
        if not np.isfinite(scores).all():
            raise ValueError("Nonfinite model scores")
        for row, user in enumerate(users):
            scores[row, seen.indices[seen.indptr[user]:seen.indptr[user + 1]]] = -np.inf
        if np.any(np.isfinite(scores).sum(axis=1) < k):
            raise ValueError("Fewer than K unseen candidates")
        # Stable full sort makes ties reproducible: smaller encoded item ID wins.
        return np.argsort(-scores, axis=1, kind="stable")[:, :k]

