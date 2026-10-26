import numpy as np
import pandas as pd
import pytest
import torch
from scipy import sparse
from models import IALS, EASE, ItemKNN, CustomTargetALS, FactorizationMachine
from models.fm import FMCore
from data_utils import matrices, train_metadata
from evaluation import ranking_metrics, evaluation_set


def test_als_update_equals_dense_weighted_least_squares():
    y = np.random.default_rng(1).normal(size=(5, 3))
    target = sparse.csr_matrix(([1., 0., -0.5], ([0, 0, 0], [0, 2, 4])), shape=(1, 5))
    confidence = np.array([8., 1., 8., 1., 8.])
    p = np.array([1., 0., 0., 0., -0.5])
    expected = np.linalg.solve(y.T @ np.diag(confidence) @ y + .7 * np.eye(3), y.T @ (confidence * p))
    actual = IALS.update_rows(target, y, 7., .7)[0]
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_als_objective_decreases_and_matches_dense():
    p = sparse.csr_matrix(([1., 0., 1., 1.], ([0, 0, 1, 2], [0, 1, 2, 3])), shape=(3, 4))
    model = IALS(factors=2, iterations=5).fit(p)
    assert np.all(np.diff(model.loss_history) <= 1e-8)
    c = np.ones(p.shape)
    co = p.tocoo()
    c[co.row, co.col] += model.alpha
    dense_loss = (c * (p.toarray() - model.score(np.arange(3))) ** 2).sum()
    dense_loss += model.regularization * ((model.users ** 2).sum() + (model.items ** 2).sum())
    np.testing.assert_allclose(model.objective(p), dense_loss, atol=1e-10)


def test_ease_matches_independent_constrained_regressions():
    x = sparse.csr_matrix([[1., 0, 1, 0], [0, 1, 1, 0], [1, 0, 0, 1], [0, 1, 0, 1]])
    model = EASE(2.).fit(x)
    a = x.toarray()
    for j in range(4):
        idx = np.arange(4) != j
        expected = np.linalg.solve(a[:, idx].T @ a[:, idx] + 2 * np.eye(3), a[:, idx].T @ a[:, j])
        np.testing.assert_allclose(model.weights[idx, j], expected, atol=1e-12)
    assert np.all(np.diag(model.weights) == 0)


def test_knn_cosine_and_direction():
    x = sparse.csr_matrix([[1., 1, 0], [1, 0, 1], [0, 1, 0]])
    model = ItemKNN(neighbors=2, shrinkage=0).fit(x)
    expected = np.array([[0, .5, 1 / np.sqrt(2)], [.5, 0, 0], [1 / np.sqrt(2), 0, 0]])
    np.testing.assert_allclose(model.similarity.toarray(), expected)
    np.testing.assert_allclose(model.score([0]), x.toarray()[[0]] @ expected)


def test_custom_changes_preference_and_retains_confidence_support():
    rows = pd.DataFrame({"user": [0, 0, 1], "item": [0, 1, 2],
                         "rating": [1., 4., 5.], "timestamp": [0, 0, 86400 * 365]})
    cutoff = 86400 * 365
    custom = CustomTargetALS.preference(rows.rating, rows.timestamp, cutoff, 365)
    np.testing.assert_allclose(custom, [-.75, .375, 1.])
    _, _, base = matrices(rows, (2, 3))
    target = sparse.csr_matrix((custom, (rows.user, rows.item)), shape=(2, 3))
    np.testing.assert_array_equal(base.indices, target.indices)
    np.testing.assert_array_equal(base.indptr, target.indptr)
    assert not np.allclose(base.data, target.data)
    with pytest.raises(ValueError):
        CustomTargetALS.preference([5], [cutoff + 1], cutoff, 365)


def test_fm_fast_formula_equals_explicit_pair_interactions():
    torch.manual_seed(10)
    um = np.array([[.2, -.3], [.7, .1]], np.float32)
    im = np.array([[.4], [.8], [-.2]], np.float32)
    m = FMCore(2, 3, um, im, 4)
    u, i = 1, 2
    vectors = torch.cat([m.user_v.weight[u:u+1], m.item_v.weight[i:i+1], m.um_v, m.im_v])
    x = torch.tensor([1, 1, *um[u], *im[i]])
    w = torch.cat([m.user_w.weight[u].reshape(1), m.item_w.weight[i].reshape(1), m.um_w, m.im_w])
    explicit = m.bias + w @ x
    for a in range(len(x)):
        for b in range(a + 1, len(x)):
            explicit = explicit + (vectors[a] @ vectors[b]) * x[a] * x[b]
    actual = m(torch.tensor([u]), torch.tensor([i]))[0]
    torch.testing.assert_close(actual, explicit)
    actual.backward()
    assert torch.isfinite(m.um_v.grad).all()


def test_metrics_hand_calculation_and_recommendation_mask():
    ndcg, recall = ranking_metrics([4, 2, 8], {2, 7})
    np.testing.assert_allclose(ndcg, (1 / np.log2(3)) / (1 + 1 / np.log2(3)))
    assert recall == .5
    x = sparse.csr_matrix(([1., 1.], ([0, 1], [0, 1])), shape=(2, 15))
    model = EASE(10).fit(x)
    recs = model.recommend([0, 1], x)
    assert len(set(recs[0])) == 10 and 0 not in recs[0] and 1 not in recs[1]
    np.testing.assert_array_equal(recs[0], np.arange(1, 11))


def test_cold_users_and_zero_positive_users_are_explicitly_excluded():
    seen = sparse.csr_matrix(([1.], ([0], [0])), shape=(2, 15))
    frame = pd.DataFrame({"user": [-1, 0, 0, 1], "item": [2, 0, 2, 3], "rating": [5, 5, 5, 2]})
    truth, audit = evaluation_set(frame, seen)
    assert truth == {0: {2}}
    assert audit["cold_user_rows"] == 1 and audit["known_users_without_relevant"] == 1


def test_fm_training_reproducible_and_score_matches_forward():
    frame = pd.DataFrame({"user": [0, 0, 1, 2], "item": [0, 1, 2, 3],
                          "rating": [5., 2., 4., 5.], "timestamp": [1, 2, 3, 4]})
    um, im = train_metadata(frame, (3, 15), 10)
    models = [FactorizationMachine(factors=3, epochs=2, batch_size=4).fit(frame, (3, 15), um, im) for _ in range(2)]
    np.testing.assert_allclose(models[0].score([0, 2]), models[1].score([0, 2]))
    with torch.no_grad():
        expected = models[0].model(torch.tensor([0, 2]), torch.tensor([1, 3])).numpy()
    np.testing.assert_allclose(models[0].score([0, 2])[[0, 1], [1, 3]], expected, atol=1e-6)
