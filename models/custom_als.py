import numpy as np
from scipy import sparse
from .ials import IALS


class CustomTargetALS(IALS):
    """Signed, recency-aware preference; confidence is identical to base iALS."""

    def __init__(self, half_life_days=365.0, **kwargs):
        super().__init__(**kwargs)
        self.half_life_days = half_life_days

    @staticmethod
    def preference(rating, timestamp, cutoff, half_life_days):
        age = (cutoff - np.asarray(timestamp)) / 86400.0
        if np.any(age < 0) or half_life_days <= 0:
            raise ValueError("Training event from the future or invalid half-life")
        signed = np.clip((np.asarray(rating) - 3.0) / 2.0, -1, 1)
        return signed * (0.5 + 0.5 * np.exp2(-age / half_life_days))

    def fit_reviews(self, reviews, shape, cutoff):
        p = self.preference(reviews.rating, reviews.timestamp, cutoff, self.half_life_days)
        targets = sparse.csr_matrix((p, (reviews.user, reviews.item)), shape=shape)
        return self.fit(targets)

