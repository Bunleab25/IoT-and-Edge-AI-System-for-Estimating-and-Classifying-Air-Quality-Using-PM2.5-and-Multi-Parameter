"""Metrics shared by the final chronological evaluation and runnable demo."""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
)


def align_probabilities(probabilities, estimator_classes, class_ids=range(4)) -> np.ndarray:
    """Place estimator probabilities into the complete ordinal class order."""
    aligned = np.zeros((len(probabilities), len(class_ids)), dtype=float)
    class_positions = {class_id: index for index, class_id in enumerate(class_ids)}
    for source_index, class_value in enumerate(estimator_classes):
        class_id = int(class_value)
        if class_id not in class_positions:
            raise ValueError(f"Estimator returned unknown class id {class_id}.")
        aligned[:, class_positions[class_id]] = probabilities[:, source_index]
    row_sums = aligned.sum(axis=1, keepdims=True)
    aligned = np.divide(aligned, row_sums, out=np.full_like(aligned, 1 / len(class_ids)), where=row_sums > 0)
    return aligned


def classification_metrics(actual, predicted, probabilities, class_ids=range(4)) -> dict:
    """Calculate aggregate, per-class, and probabilistic multiclass metrics."""
    labels = list(class_ids)
    precision, recall, f1, support = precision_recall_fscore_support(
        actual,
        predicted,
        labels=labels,
        zero_division=0,
    )
    one_hot = np.eye(len(labels), dtype=float)[np.asarray(actual, dtype=int)]
    brier_score = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
    per_class = {
        str(class_id): {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, class_id in enumerate(labels)
    }
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(actual, predicted, labels=labels, average="weighted", zero_division=0)),
        "matthews_correlation": float(matthews_corrcoef(actual, predicted)),
        "log_loss": float(log_loss(actual, probabilities, labels=labels)),
        "multiclass_brier_score": brier_score,
        "per_class": per_class,
    }


def hard_prediction_probabilities(predicted, class_ids=range(4)) -> np.ndarray:
    """Represent a deterministic baseline prediction as one-hot probabilities."""
    labels = list(class_ids)
    positions = {class_id: index for index, class_id in enumerate(labels)}
    probabilities = np.zeros((len(predicted), len(labels)), dtype=float)
    for row, class_id in enumerate(predicted):
        class_id = int(class_id)
        if class_id not in positions:
            raise ValueError(f"Baseline returned unknown class id {class_id}.")
        probabilities[row, positions[class_id]] = 1.0
    return probabilities
