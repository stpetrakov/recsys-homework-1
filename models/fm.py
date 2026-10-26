import numpy as np
import torch
from torch import nn
from .base import Recommender


class FMCore(nn.Module):
    """Second-order FM, evaluated by splitting features into user/item groups."""
    def __init__(self, n_users, n_items, user_meta, item_meta, factors):
        super().__init__()
        self.user_v = nn.Embedding(n_users, factors)
        self.item_v = nn.Embedding(n_items, factors)
        self.user_w = nn.Embedding(n_users, 1)
        self.item_w = nn.Embedding(n_items, 1)
        self.um_v = nn.Parameter(torch.randn(user_meta.shape[1], factors) * 0.03)
        self.im_v = nn.Parameter(torch.randn(item_meta.shape[1], factors) * 0.03)
        self.um_w = nn.Parameter(torch.zeros(user_meta.shape[1]))
        self.im_w = nn.Parameter(torch.zeros(item_meta.shape[1]))
        self.bias = nn.Parameter(torch.zeros(()))
        self.register_buffer("um", torch.as_tensor(user_meta, dtype=torch.float32))
        self.register_buffer("im", torch.as_tensor(item_meta, dtype=torch.float32))
        nn.init.normal_(self.user_v.weight, std=0.03)
        nn.init.normal_(self.item_v.weight, std=0.03)
        nn.init.zeros_(self.user_w.weight)
        nn.init.zeros_(self.item_w.weight)

    @staticmethod
    def group(ids, embeddings, linear, meta, meta_v, meta_w):
        v = embeddings(ids)
        x = meta[ids]
        summed = v + x @ meta_v
        squared = v.square() + x.square() @ meta_v.square()
        intercept = linear(ids).squeeze(-1) + x @ meta_w
        intercept = intercept + 0.5 * (summed.square() - squared).sum(dim=1)
        return summed, intercept

    def representations(self, users, items):
        u, bu = self.group(users, self.user_v, self.user_w, self.um, self.um_v, self.um_w)
        i, bi = self.group(items, self.item_v, self.item_w, self.im, self.im_v, self.im_w)
        return u, bu, i, bi

    def forward(self, users, items):
        u, bu, i, bi = self.representations(users, items)
        return self.bias + bu + bi + (u * i).sum(dim=1)


class FactorizationMachine(Recommender):
    def __init__(self, factors=32, epochs=8, learning_rate=0.01,
                 regularization=1e-5, negatives=2, batch_size=4096, seed=42):
        self.factors, self.epochs, self.learning_rate = factors, epochs, learning_rate
        self.regularization, self.negatives = regularization, negatives
        self.batch_size, self.seed = batch_size, seed

    def fit(self, reviews, shape, user_meta, item_meta):
        torch.manual_seed(self.seed)
        torch.use_deterministic_algorithms(True)
        rng = np.random.default_rng(self.seed)
        self.model = FMCore(*shape, user_meta, item_meta, self.factors)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,
                                     weight_decay=self.regularization)
        users = reviews.user.to_numpy(dtype=np.int64)
        items = reviews.item.to_numpy(dtype=np.int64)
        labels = (reviews.rating.to_numpy() >= 4).astype(np.float32)
        positives = users[labels == 1]
        seen_keys = np.sort(users * shape[1] + items)
        all_ids = np.arange(shape[1])
        if any(len(g) >= shape[1] for _, g in reviews.groupby("user")):
            raise ValueError("A user has no unseen negatives")
        self.loss_history = []
        for _ in range(self.epochs):
            neg_users = np.repeat(positives, self.negatives)
            neg_items = rng.integers(shape[1], size=len(neg_users))
            bad = np.isin(neg_users * shape[1] + neg_items, seen_keys)
            while bad.any():
                neg_items[bad] = rng.choice(all_ids, size=bad.sum())
                bad = np.isin(neg_users * shape[1] + neg_items, seen_keys)
            u = torch.from_numpy(np.concatenate([users, neg_users]))
            i = torch.from_numpy(np.concatenate([items, neg_items]))
            y = torch.from_numpy(np.concatenate([labels, np.zeros(len(neg_users), np.float32)]))
            order = rng.permutation(len(y))
            total = 0.0
            for start in range(0, len(order), self.batch_size):
                ix = order[start:start + self.batch_size]
                optimizer.zero_grad()
                logits = self.model(u[ix], i[ix])
                loss = nn.functional.binary_cross_entropy_with_logits(logits, y[ix])
                loss.backward()
                optimizer.step()
                total += loss.item() * len(ix)
            self.loss_history.append(total / len(y))
        self.model.eval()
        with torch.no_grad():
            u, bu, i, bi = self.model.representations(torch.arange(shape[0]), torch.arange(shape[1]))
            self.user_vectors, self.user_bias = u.numpy(), bu.numpy()
            self.item_vectors, self.item_bias = i.numpy(), bi.numpy()
            self.bias = self.model.bias.item()
        return self

    def score(self, users):
        return (self.user_vectors[users] @ self.item_vectors.T +
                self.user_bias[users, None] + self.item_bias[None, :] + self.bias)
