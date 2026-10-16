"""Classification metrics in manifest class order, with explicit undefined AUCs."""

import numpy as np


def _binary_auc(positive: np.ndarray, scores: np.ndarray) -> float | None:
    positives = int(positive.sum())
    negatives = len(positive) - positives
    if not positives or not negatives:
        return None
    order = np.argsort(scores, kind="stable")
    _, starts, counts = np.unique(scores[order], return_index=True, return_counts=True)
    # Average one-based ranks give tied positive/negative pairs half credit.
    ranks = np.repeat(starts + (counts + 1) / 2, counts)
    rank_sum = float(ranks[positive[order]].sum())
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def classification_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    image_keys: list[str],
    class_names: tuple[str, ...],
) -> dict:
    targets, probabilities = np.asarray(targets), np.asarray(probabilities)
    classes = len(class_names)
    if classes < 2 or len(set(class_names)) != classes:
        raise ValueError("Expected at least two unique classes")
    if (
        targets.ndim != 1
        or not len(targets)
        or not np.issubdtype(targets.dtype, np.integer)
    ):
        raise ValueError("Expected a nonempty vector of integer targets")
    if (
        probabilities.shape != (len(targets), classes)
        or not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
        or not np.allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-6)
    ):
        raise ValueError(
            "Expected finite class probabilities with each row summing to one"
        )
    if (targets < 0).any() or (targets >= classes).any():
        raise ValueError("Target is outside the class map")
    targets = targets.astype(np.int64, copy=False)
    if len(image_keys) != len(targets) or len(set(image_keys)) != len(targets):
        raise ValueError("Expected one unique image key per target")
    predicted = probabilities.argmax(axis=1)
    confusion = np.bincount(
        targets * classes + predicted, minlength=classes**2
    ).reshape(classes, classes)
    per_class = {}
    for index, name in enumerate(class_names):
        correct = int(confusion[index, index])
        support, predicted_count = (
            int(confusion[index].sum()),
            int(confusion[:, index].sum()),
        )
        per_class[name] = {
            "support": support,
            "precision": correct / predicted_count if predicted_count else 0.0,
            "recall": correct / support if support else 0.0,
            "f1": 2 * correct / (support + predicted_count)
            if support + predicted_count
            else 0.0,
            "roc_auc": _binary_auc(targets == index, probabilities[:, index]),
        }
    aucs = [row["roc_auc"] for row in per_class.values()]
    if classes == 2:
        auc = aucs[1]
    elif all(value is not None for value in aucs):
        auc = float(np.mean(aucs))
    else:
        auc = None
    confidence = probabilities[np.arange(len(targets)), predicted]
    mistakes = sorted(
        np.flatnonzero(targets != predicted),
        key=lambda i: (-confidence[i], image_keys[i]),
    )[:16]
    return {
        "samples": len(targets),
        "accuracy": float(np.trace(confusion) / len(targets)),
        "per_class": per_class,
        "macro_f1": float(np.mean([row["f1"] for row in per_class.values()])),
        "roc_auc": auc,
        "roc_auc_method": "binary" if classes == 2 else "macro_one_vs_rest",
        "positive_class": class_names[1] if classes == 2 else None,
        "zero_division": 0,
        "confusion_matrix": {
            "class_names": list(class_names),
            "rows": "true",
            "columns": "predicted",
            "counts": confusion.tolist(),
        },
        "confident_mistakes": [
            {
                "image_key": image_keys[i],
                "true_label": class_names[targets[i]],
                "predicted_label": class_names[predicted[i]],
                "confidence": float(confidence[i]),
            }
            for i in mistakes
        ],
    }
