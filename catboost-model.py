# CatBoost model for credit card fraud detection

from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


DATA_FILE = Path(__file__).resolve().parent / "creditcard.csv"
TARGET_COLUMN = "Class"
TEST_SIZE = 0.2     # Batch 80/20
RANDOM_STATE = 42   # pseudo-random, keep the same mixe


# - Load the data
# The CSV file must be in the same folder as this script
if not DATA_FILE.exists():
    raise FileNotFoundError(
        "creditcard.csv was not found. Run data-import.py before this script."
    )

df = pd.read_csv(DATA_FILE) # DataFrame in type : pandas.core.frame.DataFrame


# - Show basic information about the data
"""
print("Dataset shape:", df.shape)
print("\nFirst rows:")
print(df.head())

print("\nTarget distribution:")
print(df[TARGET_COLUMN].value_counts())

print("\nTarget distribution in percent:")
print(df[TARGET_COLUMN].value_counts(normalize=True) * 100)
"""

# - Split the data into inputs and target
# X contains the columns used to make a prediction
# y contains the answer we want to predict
X = df.drop(columns=TARGET_COLUMN)  # All except Class
y = df[TARGET_COLUMN]               # Only Class


# - Split the data into train and test sets
# stratify=y keeps the same fraud ratio in train and test => here 394 + 98 = 492 fraudulent transactions
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

"""
print("\n\033[94mTrain target distribution:\033[0m")
print(y_train.value_counts())

print("\n\033[94mTest target distribution:\033[0m")
print(y_test.value_counts())
"""

# - Creating the model
# CatBoost trains many decision trees one after the other
# Each new tree tries to correct the errors made by the previous trees
# class_weights helps because fraud cases are rare
# iterations = number of boosting steps
# depth = max depth of each tree
model = CatBoostClassifier(
    iterations=300,
    learning_rate=0.09,
    depth=6,
    loss_function="Logloss",
    eval_metric="AUC",
    class_weights=[1, 40],
    random_seed=RANDOM_STATE,
    thread_count=-1,
    allow_writing_files=False,
    verbose=False,
)


# - Train the model
# The model only learns from the train data => 80% of the data set
model.fit(X_train, y_train)


# - Test the model
# y_pred contains the predicted class: 0 or 1
# y_score contains the fraud probability

#y_pred = model.predict(X_test) # Only predicted [0, 0, 1, ...]  => here proba >= 0.5

y_score = model.predict_proba(X_test)[:, 1] # All lines and only the second column => proba fraud [[0.99, 0.01], ...] => [0.01]
THRESHOLD = 0.4
y_pred = (y_score >= THRESHOLD).astype(int) # 'astype(int)' change true/false in 1/0

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

# ROC-AUC uses the fraud probabilities, not only the final 0/1 prediction
# It checks if fraud cases usually get a higher score than normal cases
roc_auc = roc_auc_score(y_test, y_score)

# PR-AUC means Precision-Recall AUC.
# It is very useful when one class is rare, like fraud in this dataset
pr_auc = average_precision_score(y_test, y_score)

print("ROC-AUC:", roc_auc)
print("PR-AUC:", pr_auc)
