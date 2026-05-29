# Credit Card Fraud Detection

Projet de test de plusieurs modeles de machine learning sur le dataset Kaggle
`mlg-ulb/creditcardfraud`.

Le dataset est tres desequilibre :

- 284 807 transactions
- 492 fraudes
- environ 0.172% de fraude

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python data-import.py
```

Le fichier `creditcard.csv` est telecharge localement a cote de `data-import.py`.

## Test effectues

1. Logistic Regression
2. Random Forest
3. Gradient Boosting
4. XGBoost
5. LightGBM
6. Isolation Forest
7. KNN
8. SVM

Commencer par la Logistic Regression permet d'avoir une baseline simple et
rapide. KNN et SVM sont plutot a tester plus tard,
car ils peuvent devenir lents sur un dataset de cette taille.

## Fichiers

- `data-import.py` : telechargement du dataset
- `logistic-regression.py`
- `random-forest.py`
- `gradient-boosting.py` : utilise `HistGradientBoostingClassifier`, une version plus rapide du gradient boosting sklearn
- `xgboost-model.py`
- `lightgbm-model.py`
- `knn.py`
- `svm.py`
- `isolation-forest.py`

## Metriques compare

Comme les fraudes sont rares, l'accuracy seule n'est pas suffisante.

Suivi pour chaque modele :

| Metrique | Description | Valeur haute | Valeur basse |
| --- | --- | --- | --- |
| Precision | Parmi les transactions predites comme fraude, combien sont vraiment des fraudes. | Peu de fausses alertes. | Beaucoup de fausses alertes. |
| Recall | Parmi les vraies fraudes, combien sont trouvees par le modele. | Peu de fraudes ratees. | Beaucoup de fraudes ratees. |
| F1-score | Equilibre entre precision et recall. | Bon compromis. | Precision ou recall trop faible. |
| Accuracy | Part totale de bonnes predictions. | Beaucoup de bonnes predictions. | Beaucoup d'erreurs, mais peut etre trompeuse si les classes sont desequilibrees. |
| ROC-AUC | Capacite du modele a donner un score plus haut aux fraudes qu'aux transactions normales. | Bon classement global. | Mauvais classement. |
| PR-AUC | Qualite du compromis precision/recall. | Bon resultat sur la classe rare. | Mauvais compromis pour detecter les fraudes. |
| FP | False positives : transactions normales predites comme fraude. | Beaucoup de fausses alertes. | Peu de fausses alertes. |
| FN | False negatives : fraudes predites comme normales. | Beaucoup de fraudes ratees. | Peu de fraudes ratees. |
| Confusion matrix | Detail des bonnes et mauvaises predictions. | Permet de voir les bonnes predictions. | Permet de voir les fausses alertes et les fraudes ratees. |

## Resultats

| Modele | Precision | Recall | F1-score | Accuracy | ROC-AUC | PR-AUC | FP | FN | Notes | Conclusion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.4372 | 0.8878 | 0.5859 | 0.9978 | 0.9690 | 0.7488 | 112 | 11 | `StandardScaler`, `C=0.1`, `class_weight={0: 1, 1: 25}`, `threshold=0.3`, `max_iter=2000`, grille principale : 24 modeles entraines, 120 combinaisons evaluees |  |
| Random Forest | 0.9130 | 0.8571 | 0.8842 | 0.9996 | 0.9529 | 0.8639 | 8 | 14 | `StandardScaler`, `n_estimators=100`, `class_weight={0: 1, 1: 25}`, `threshold=0.3`, tests partiels : 6 modeles entraines, 48 combinaisons evaluees |  |
| Gradient Boosting | 0.8889 | 0.8163 | 0.8511 | 0.9995 | 0.9792 | 0.8540 | 10 | 18 | `StandardScaler`, `HistGradientBoostingClassifier`, `learning_rate=0.05`, `max_iter=100`, `max_leaf_nodes=15`, `min_samples_leaf=20`, `class_weight={0: 1, 1: 5}`, `threshold=0.6`, 144 modeles entraines, 1296 combinaisons evaluees | Version histogram du gradient boosting sklearn car version de base trop lente |
| XGBoost | 0.9222 | 0.8469 | 0.8830 | 0.9996 | 0.9728 | 0.8874 | 7 | 15 | `StandardScaler`, `XGBClassifier`, `n_estimators=200`, `learning_rate=0.2`, `max_depth=5`, `scale_pos_weight=25`, `threshold=0.6`, 72 modeles entraines, 648 combinaisons evaluees | Meilleur PR-AUC actuel |
| LightGBM | 0.9111 | 0.8367 | 0.8723 | 0.9996 | 0.9717 | 0.8584 | 8 | 16 | `LGBMClassifier`, `n_estimators=200`, `learning_rate=0.03`, `max_depth=4`, `num_leaves=15`, `min_child_samples=100`, `scale_pos_weight=1`, `threshold=0.3`, 540 modeles entraines, 4968 combinaisons evaluees | Bon modele, mais moins bon que Random Forest et XGBoost sur F1-score et PR-AUC |
| Isolation Forest | 0.3509 | 0.4082 | 0.3774 | 0.9977 | 0.9612 | 0.3292 | 74 | 58 | `StandardScaler`, `IsolationForest`, `n_estimators=200`, `max_samples=4096`, `max_features=0.5`, `top_anomaly_rate=0.002`, non supervise, 48 modeles entraines, 288 combinaisons evaluees | Detecteur d'anomalies interessant, mais beaucoup moins performant que les modeles supervises sur ce dataset |
| KNN | | | | | | | | | | |
| SVM | | | | | | | | | | |
