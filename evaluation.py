import numpy as np


def evaluation_set(holdout, seen):
    known = holdout[holdout.user >= 0]
    relevant = known[known.rating >= 4]
    truth = {}
    for user, group in relevant.groupby("user"):
        old = set(seen.indices[seen.indptr[user]:seen.indptr[user + 1]])
        items = set(group.item) - old
        if items and seen.shape[1] - len(old) >= 10:
            truth[int(user)] = items
    return truth, {"holdout_rows": len(holdout), "cold_user_rows": int((holdout.user < 0).sum()),
                   "known_users_without_relevant": int(known.user.nunique() - len(truth)),
                   "evaluated_users": len(truth), "relevant_pairs": sum(map(len, truth.values()))}


def ranking_metrics(recommendations, relevant):
    hits = np.array([item in relevant for item in recommendations], dtype=float)
    discounts = 1 / np.log2(np.arange(2, len(recommendations) + 2))
    recall = hits.sum() / len(relevant)
    ndcg = (hits * discounts).sum() / discounts[:min(len(relevant), len(recommendations))].sum()
    return float(ndcg), float(recall)


def evaluate(model, truth, seen, batch_size=128):
    users = np.array(sorted(truth))
    if len(users) == 0:
        raise ValueError("Empty evaluation cohort")
    per_user, all_recs = [], []
    for start in range(0, len(users), batch_size):
        batch = users[start:start + batch_size]
        recs = model.recommend(batch, seen, k=10)
        for user, rec in zip(batch, recs):
            per_user.append(ranking_metrics(rec, truth[user]))
        all_recs.append(recs)
    values = np.asarray(per_user)
    return {"ndcg@10": float(values[:, 0].mean()), "recall@10": float(values[:, 1].mean()),
            "coverage@10": float(len(np.unique(np.concatenate(all_recs))) / seen.shape[1]),
            "users": len(users)}, values


def paired_bootstrap_delta(first, second, seed=42, samples=2000):
    """User-level paired percentile bootstrap; descriptive, not multiplicity-corrected."""
    delta = np.asarray(first) - np.asarray(second)
    rng = np.random.default_rng(seed)
    means = np.array([delta[rng.integers(len(delta), size=len(delta))].mean() for _ in range(samples)])
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {"delta": float(delta.mean()), "ci95_low": float(lo), "ci95_high": float(hi)}

