# Helpers for experiment scripts
#
# The files in experiments/ often test many thresholds or many weight
# combinations. Without helper functions, each experiment would repeat the same
# metrics code.
#
# This file keeps the common experiment logic in one place:
# - convert model scores into class predictions
# - calculate metrics
# - print compact result rows
# - select the best result according to a simple rule

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
    # y_score contains one score per transaction.
    # A higher score means "more likely to be fraud".
    #
    # threshold is the cut-off used to convert scores into classes:
    # - score >= threshold => fraud (1)
    # - score < threshold  => normal (0)
    y_pred = (y_score >= threshold).astype(int)

    # Confusion matrix layout for binary classification:
    #
    # [[TN, FP],
    #  [FN, TP]]
    #
    # TN = true negatives  = normal transactions predicted as normal
    # FP = false positives = normal transactions predicted as fraud
    # FN = false negatives = fraud transactions predicted as normal
    # TP = true positives  = fraud transactions predicted as fraud
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Return all useful metrics in a dictionary.
    # Experiments can store many dictionaries in a list, then search the best one.
    return {
        "threshold": threshold,

        # Precision:
        # among predicted frauds, how many are real frauds.
        # High precision means few false positives.
        "precision": precision_score(y_test, y_pred, zero_division=0),

        # Recall:
        # among real frauds, how many were found.
        # High recall means few false negatives.
        "recall": recall_score(y_test, y_pred, zero_division=0),

        # F1-score:
        # balance between precision and recall.
        "f1": f1_score(y_test, y_pred, zero_division=0),

        # Accuracy:
        # total correct predictions / total predictions.
        # Be careful: accuracy can be misleading when fraud is very rare.
        "accuracy": accuracy_score(y_test, y_pred),

        # ROC-AUC and PR-AUC use the raw score, not the final 0/1 prediction.
        # This is useful because they evaluate score quality across thresholds.
        "roc_auc": roc_auc_score(y_test, y_score),
        "pr_auc": average_precision_score(y_test, y_score),

        # Store confusion matrix values as normal Python int values.
        # This makes printing and README copy/paste easier.
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "tn": int(tn),
    }


def print_row(name, metrics):
    # Print one compact result line.
    #
    # The formatting keeps columns aligned enough to compare several models
    # directly in the terminal.
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
    # Select the best row by F1-score.
    #
    # If two rows have the same F1:
    # - prefer the one with higher recall
    # - then prefer the one with fewer false positives
    #
    # This matches the project goal:
    # fraud recall is important, but false positives still matter.
    return max(rows, key=lambda row: (row["f1"], row["recall"], -row["fp"]))


def best_by_low_fn(rows, max_fp=50):
    # Select the row with the lowest false negatives,
    # but only among rows where false positives stay under max_fp.
    #
    # This avoids choosing an aggressive threshold that catches many frauds
    # but creates too many false alerts.
    valid_rows = [row for row in rows if row["fp"] <= max_fp]

    # If no row respects the FP limit, return None.
    # The calling script can then skip this result.
    if not valid_rows:
        return None

    # Sort priority:
    # 1. lowest FN
    # 2. lowest FP
    # 3. highest F1
    return min(valid_rows, key=lambda row: (row["fn"], row["fp"], -row["f1"]))
