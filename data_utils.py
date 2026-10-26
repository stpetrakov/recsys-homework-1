"""Download, deterministic cleaning, temporal splits, and train-only metadata."""
from pathlib import Path
import gzip
import hashlib
import json
import urllib.request
import numpy as np
import pandas as pd
from scipy import sparse

DATA_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall/Digital_Music_5.json.gz"
DATA_SHA256 = "703f60f74bf1967a75cbb3033532172eeb21139d4df04ec5fe44a7eb8d7fde58"
TRAIN_END = 1451347200  # 2015-12-29 00:00:00 UTC
VALID_END = 1475020800  # 2016-09-28 00:00:00 UTC
CATALOG_SIZE = 6000


def download(path="data/Digital_Music_5.json.gz"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(".download")
        with urllib.request.urlopen(DATA_URL, timeout=180) as response, temporary.open("wb") as out:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
        temporary.replace(path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != DATA_SHA256:
        raise ValueError(f"Dataset checksum mismatch: {actual}")
    return path


def load_data(path="data/Digital_Music_5.json.gz"):
    rows = []
    with gzip.open(download(path), "rt") as f:
        for line in f:
            d = json.loads(line)
            # Do not retain reviewer names, free-text reviews, or other unused fields.
            rows.append((d.get("reviewerID"), d.get("asin"), d.get("overall"), d.get("unixReviewTime")))
    raw = pd.DataFrame(rows, columns=["uid", "iid", "rating", "timestamp"])
    audit = {"raw_rows": len(raw), "missing": raw.isna().sum().to_dict(),
             "repeated_user_item_rows": int(raw.duplicated(["uid", "iid"]).sum()),
             "exact_tuple_duplicates": int(raw.duplicated().sum())}
    cleaned = raw.dropna().copy()
    cleaned = cleaned[cleaned.rating.between(1, 5) & (cleaned.timestamp > 0)]
    # Earliest event is the first recommendation opportunity; stable source order breaks same-day ties.
    cleaned = cleaned.sort_values("timestamp", kind="stable").drop_duplicates(["uid", "iid"], keep="first")
    audit["clean_rows"] = len(cleaned)
    pretrain = cleaned[cleaned.timestamp < TRAIN_END]
    popularity = pretrain.groupby("iid").size().rename("n").reset_index()
    catalog = popularity.sort_values(["n", "iid"], ascending=[False, True]).head(CATALOG_SIZE).iid
    selected = cleaned[cleaned.iid.isin(catalog)].copy()
    item_map = {v: n for n, v in enumerate(sorted(catalog))}
    user_map = {v: n for n, v in enumerate(sorted(selected.loc[selected.timestamp < TRAIN_END, "uid"].unique()))}
    selected["user"] = selected.uid.map(user_map).fillna(-1).astype(int)
    selected["item"] = selected.iid.map(item_map).astype(int)
    train = selected[selected.timestamp < TRAIN_END].copy()
    valid = selected[(selected.timestamp >= TRAIN_END) & (selected.timestamp < VALID_END)].copy()
    test = selected[selected.timestamp >= VALID_END].copy()
    audit.update(selected_rows=len(selected), selected_users=selected.uid.nunique(),
                 selected_items=selected.iid.nunique(), train_rows=len(train),
                 validation_rows=len(valid), test_rows=len(test), model_users=len(user_map),
                 sparsity=1 - len(selected) / (selected.uid.nunique() * selected.iid.nunique()),
                 first_date=str(pd.to_datetime(selected.timestamp.min(), unit="s", utc=True)),
                 last_date=str(pd.to_datetime(selected.timestamp.max(), unit="s", utc=True)))
    for label, left, right in [("validation", TRAIN_END, VALID_END), ("test", VALID_END, float("inf"))]:
        period = cleaned[(cleaned.timestamp >= left) & (cleaned.timestamp < right)]
        audit[label + "_all_catalog_relevant"] = int((period.rating >= 4).sum())
        audit[label + "_outside_catalog_relevant"] = int(((period.rating >= 4) & ~period.iid.isin(catalog)).sum())
    assert 100_000 <= len(selected) <= 2_000_000
    assert selected.uid.nunique() >= 1000 and selected.iid.nunique() >= 1000
    assert train.timestamp.max() < valid.timestamp.min() <= valid.timestamp.max() < test.timestamp.min()
    assert not selected.duplicated(["uid", "iid"]).any()
    return selected, train, valid, test, (len(user_map), len(item_map)), audit


def matrices(reviews, shape):
    indices = (reviews.user.to_numpy(), reviews.item.to_numpy())
    seen = sparse.csr_matrix((np.ones(len(reviews)), indices), shape=shape)
    targets = sparse.csr_matrix(((reviews.rating.to_numpy() >= 4).astype(float), indices), shape=shape)
    positive = targets.copy()
    positive.eliminate_zeros()
    return seen, positive, targets


def train_metadata(reviews, shape, cutoff):
    """Three historical descriptors for each entity, fixed at recommendation cutoff.

    Numeric scalers fit on training entities, never validation/test. Values are
    log1p(review count), smoothed mean rating, log1p(days since last review).
    """
    result = []
    global_mean = float(reviews.rating.mean())
    for key, size in zip(["user", "item"], shape):
        stats = reviews.groupby(key).agg(n=("rating", "size"), total=("rating", "sum"), last=("timestamp", "max"))
        stats = stats.reindex(np.arange(size))
        count = stats.n.fillna(0).to_numpy()
        mean = (stats.total.fillna(0).to_numpy() + 5 * global_mean) / (count + 5)
        age = (cutoff - stats["last"].fillna(reviews.timestamp.min()).to_numpy()) / 86400
        assert np.all(age >= 0)
        values = np.column_stack([np.log1p(count), mean, np.log1p(age)])
        values = (values - values.mean(axis=0)) / np.maximum(values.std(axis=0), 1e-8)
        result.append(values.astype(np.float32))
    return result

