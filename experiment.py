"""Single protocol shared by the notebook and command-line experiment."""
import gc
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from data_utils import load_data, matrices, train_metadata, TRAIN_END, VALID_END
from evaluation import evaluation_set, evaluate, paired_bootstrap_delta
from models import ItemKNN, IALS, EASE, FactorizationMachine, CustomTargetALS

SEED = 42
PRIMARY_METRIC = "ndcg@10"
SEARCH = {
    "ItemKNN": [{"neighbors": k, "shrinkage": s} for k in [30, 100] for s in [0.0, 20.0]],
    "iALS": [{"factors": f, "regularization": r, "alpha": 20.0, "iterations": 10, "seed": SEED}
             for f in [16, 32] for r in [1.0, 10.0]],
    "EASE": [{"regularization": r} for r in [10.0, 100.0, 1000.0]],
    "FM": [{"factors": f, "regularization": r, "epochs": 10, "learning_rate": 0.01,
            "negatives": 2, "batch_size": 4096, "seed": SEED} for f in [16, 32] for r in [1e-5, 1e-3]],
    "CustomALS": [{"factors": f, "regularization": r, "alpha": 20.0, "iterations": 10,
                   "half_life_days": 365.0, "seed": SEED} for f in [16, 32] for r in [1.0, 10.0]],
}


def fit_model(name, params, reviews, shape, cutoff, metadata=True):
    seen, positive, targets = matrices(reviews, shape)
    if name == "ItemKNN":
        return ItemKNN(**params).fit(positive)
    if name == "EASE":
        return EASE(**params).fit(positive)
    if name == "iALS":
        return IALS(**params).fit(targets)
    if name == "CustomALS":
        return CustomTargetALS(**params).fit_reviews(reviews, shape, cutoff)
    if name == "FM":
        um, im = train_metadata(reviews, shape, cutoff)
        if not metadata:
            um, im = np.zeros_like(um), np.zeros_like(im)
        return FactorizationMachine(**params).fit(reviews, shape, um, im)
    raise ValueError(name)


def select_parameters(train, valid, shape, output="results"):
    Path(output).mkdir(exist_ok=True)
    torch.set_num_threads(2)
    np.random.seed(SEED)
    seen = matrices(train, shape)[0]
    truth, info = evaluation_set(valid, seen)
    records, best = [], {}
    with threadpool_limits(limits=2):
        for name, choices in SEARCH.items():
            best_metric = -np.inf
            for params in choices:
                started = perf_counter()
                model = fit_model(name, params, train, shape, TRAIN_END)
                fit_seconds = perf_counter() - started
                metrics, _ = evaluate(model, truth, seen)
                record = {"model": name, "params": params, "fit_seconds": fit_seconds, **metrics}
                records.append(record)
                Path(output, "validation_progress.json").write_text(json.dumps(records, ensure_ascii=False, indent=2))
                print(name, params, metrics, flush=True)
                if metrics[PRIMARY_METRIC] > best_metric:
                    best_metric, best[name] = metrics[PRIMARY_METRIC], params
                del model
                gc.collect()
    # Selection is persisted BEFORE the final test is touched.
    payload = {"seed": SEED, "primary_metric": PRIMARY_METRIC, "search": SEARCH,
               "best": best, "validation": records, "validation_cohort": info}
    Path(output, "selection.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def final_comparison(train, valid, test, shape, selection, output="results"):
    torch.set_num_threads(2)
    fit = pd.concat([train, valid[valid.user >= 0]], ignore_index=True)
    seen = matrices(fit, shape)[0]
    truth, info = evaluation_set(test, seen)
    records, per_user, curves, examples = [], {}, {}, {}
    recipes = [(name, name, params, True) for name, params in selection["best"].items()]
    # Predeclared controls: hold the selected base settings fixed.
    recipes += [("FM_ID_only", "FM", selection["best"]["FM"], False),
                ("CustomALS_matched", "CustomALS", {**selection["best"]["iALS"], "half_life_days": 365.0}, True)]
    with threadpool_limits(limits=2):
        for label, name, params, metadata in recipes:
            started = perf_counter()
            model = fit_model(name, params, fit, shape, VALID_END, metadata)
            fit_seconds = perf_counter() - started
            started = perf_counter()
            metrics, values = evaluate(model, truth, seen)
            records.append({"model": label, "params": params, "fit_seconds": fit_seconds,
                            "evaluation_seconds": perf_counter() - started, **metrics})
            per_user[label] = values
            sample_users = np.array(sorted(truth)[:3])
            examples[label] = model.recommend(sample_users, seen).tolist()
            if hasattr(model, "loss_history"):
                curves[label] = model.loss_history
            print("TEST", label, metrics, flush=True)
            del model
            gc.collect()
    comparisons = {
        "FM minus ID-only (matched settings)": paired_bootstrap_delta(per_user["FM"][:, 0], per_user["FM_ID_only"][:, 0]),
        "Custom minus iALS (matched settings)": paired_bootstrap_delta(per_user["CustomALS_matched"][:, 0], per_user["iALS"][:, 0]),
    }
    payload = {"test": records, "test_cohort": info, "paired_ndcg_differences": comparisons,
               "training_curves": curves, "top10_encoded_examples": examples}
    Path(output, "final.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    pd.DataFrame(records).to_csv(Path(output, "test_metrics.csv"), index=False)
    return payload


if __name__ == "__main__":
    _, train, valid, test, shape, audit = load_data()
    Path("results").mkdir(exist_ok=True)
    Path("results/data_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    selection = select_parameters(train, valid, shape)
    final_comparison(train, valid, test, shape, selection)
