# Common helpers for model scripts

from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


DATA_FILE = Path(__file__).resolve().parents[1] / "creditcard.csv"
TARGET_COLUMN = "Class"
TEST_SIZE = 0.2     # Batch 80/20
RANDOM_STATE = 42   # pseudo-random, keep the same mixe


def load_train_test_data(show_info=False):
    # - Load the data
    # The CSV file must be in the same folder as the scripts
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "creditcard.csv was not found. Run data-import.py before this script."
        )

    df = pd.read_csv(DATA_FILE) # DataFrame in type : pandas.core.frame.DataFrame

    if show_info:
        print("Dataset shape:", df.shape)
        print("\nFirst rows:")
        print(df.head())

        print("\nTarget distribution:")
        print(df[TARGET_COLUMN].value_counts())

        print("\nTarget distribution in percent:")
        print(df[TARGET_COLUMN].value_counts(normalize=True) * 100)

    # - Split the data into inputs and target
    # X contains the columns used to make a prediction
    # y contains the answer we want to predict
    X = df.drop(columns=TARGET_COLUMN)  # All except Class
    y = df[TARGET_COLUMN]               # Only Class

    # - Split the data into train and test sets
    # stratify=y keeps the same fraud ratio in train and test => here 394 + 98 = 492 fraudulent transactions
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def show_results(y_test, y_score, threshold=None, y_pred=None):
    # - Test the model
    # y_pred contains the predicted class: 0 or 1
    # y_score contains the score used for ROC-AUC and PR-AUC
    if y_pred is None:
        if threshold is None:
            raise ValueError("threshold is needed when y_pred is not given.")
        y_pred = (y_score >= threshold).astype(int) # 'astype(int)' change true/false in 1/0

    # - Show the results
    # Accuracy is not enough because fraud cases are very rare
    # Show accuracy, precision, recall, F1-score, ROC-AUC and PR-AUC
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    accuracy = accuracy_score(y_test, y_pred)

    print("\n\033[92m=== Results ===\033[0m")

    print("\n\033[94mConfusion matrix:\033[0m")
    print(cm)
    print("True negatives:", tn)
    print("False positives:", fp)
    print("False negatives:", fn)
    print("True positives:", tp)

    # Classification report compares the real classes with the predicted classes
    # precision = when the model predicts a class, how often it is correct
    # recall = among the real examples of a class, how many the model finds
    # f1-score = balance between precision and recall
    # support = number of real examples for each class

    # accuracy = correct predictions / all predictions
    print("\nAccuracy:", accuracy)

    print("\n\033[94mClassification report:\033[0m")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Fraud"], digits=4)) # compare real with predict with 4 digits after 0.

    # ROC-AUC uses the scores, not only the final 0/1 prediction
    # It checks if fraud cases usually get a higher score than normal cases
    roc_auc = roc_auc_score(y_test, y_score)

    # PR-AUC means Precision-Recall AUC.
    # It is very useful when one class is rare, like fraud in this dataset
    pr_auc = average_precision_score(y_test, y_score)

    print("ROC-AUC:", roc_auc)
    print("PR-AUC:", pr_auc)

    return {
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }
