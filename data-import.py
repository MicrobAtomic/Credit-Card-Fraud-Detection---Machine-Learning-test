"""
 Credit Card Fraud Detection - ULB from Kaggle website
 -> 284'807 transactions / 492 fraudulent transactions -> 0.172%

 use kagglesdk 0.1.23 => last version break the import
 pip install kagglehub==1.0.1 kagglesdk==0.1.23 // pip install -r requirements.txt

 Time => number of seconds elapsed between this transaction and the first transaction in the dataset
 V1, V2, ..., V28 => may be result of a PCA Dimensionality reduction to protect user identities and sensitive freatures
 Amount => transaction amount 0 - 25.7k
 Class => 1 for fraudulent transactions, 0 otherwise
"""

from pathlib import Path
import kagglehub

DATASET_HANDLE = "mlg-ulb/creditcardfraud"
DATASET_FILE = "creditcard.csv"
PROJECT_DIR = Path(__file__).resolve().parent

csv_path = kagglehub.dataset_download(
    DATASET_HANDLE,
    path=DATASET_FILE,
    output_dir=str(PROJECT_DIR),
)

print("Path to dataset file:", csv_path)
