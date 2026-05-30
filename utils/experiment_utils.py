# Helpers for experiment scripts

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def get_metrics(y_test, y_score, threshold):
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    return {
        "threshold": threshold,
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_score),
        "pr_auc": average_precision_score(y_test, y_score),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "tn": int(tn),
    }


def print_row(name, metrics):
    print(
        f"{name:<14}"
        f" threshold={metrics['threshold']:<4}"
        f" FP={metrics['fp']:<4}"
        f" FN={metrics['fn']:<4}"
        f" precision={metrics['precision']:.4f}"
        f" recall={metrics['recall']:.4f}"
        f" f1={metrics['f1']:.4f}"
        f" pr_auc={metrics['pr_auc']:.4f}"
    )


def best_by_f1(rows):
    return max(rows, key=lambda row: (row["f1"], row["recall"], -row["fp"]))


def best_by_low_fn(rows, max_fp=50):
    valid_rows = [row for row in rows if row["fp"] <= max_fp]
    if not valid_rows:
        return None
    return min(valid_rows, key=lambda row: (row["fn"], row["fp"], -row["f1"]))
