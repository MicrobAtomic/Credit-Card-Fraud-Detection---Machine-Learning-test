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
python models/random-forest.py
```

Le fichier `creditcard.csv` est telecharge localement a cote de `data-import.py`.

## Test effectues

1. Logistic Regression
2. Random Forest
3. Gradient Boosting
4. XGBoost
5. LightGBM
6. CatBoost
7. Isolation Forest
8. KNN
9. SVM

Commencer par la Logistic Regression permet d'avoir une baseline simple et
rapide. KNN et SVM sont plutot a tester plus tard,
car ils peuvent devenir lents sur un dataset de cette taille.

## Fichiers

- `data-import.py` : telechargement du dataset
- `utils/model_utils.py` : chargement des donnees, split train/test et affichage des resultats
- `models/logistic-regression.py`
- `models/random-forest.py`
- `models/gradient-boosting.py` : utilise `HistGradientBoostingClassifier`, une version plus rapide du gradient boosting sklearn
- `models/xgboost-model.py`
- `models/lightgbm-model.py`
- `models/catboost-model.py`
- `models/knn.py`
- `models/svm.py`
- `models/isolation-forest.py`
- `experiments/` : tests avances avant de les ajouter au tableau principal

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

Le tableau est trie selon un compromis FN/FP : on privilegie les faux negatifs bas,
mais un petit gain en FN ne justifie pas une forte hausse des faux positifs.

| Modele | Precision | Recall | F1-score | Accuracy | ROC-AUC | PR-AUC | FP | FN | Notes | Conclusion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Random Forest | 0.9247 | 0.8776 | 0.9005 | 0.9997 | 0.9581 | 0.8678 | 7 | 12 | `StandardScaler`, `n_estimators=90`, `class_weight={0: 1, 1: 15}`, `threshold=0.3`, batch partiel : 36 modeles entraines, 540 combinaisons evaluees; option plus agressive : FP 38, FN 10 avec `n_estimators=60`, `class_weight=balanced`, `max_depth=24`, `min_samples_leaf=2`, `max_features=sqrt`, `threshold=0.1` | Meilleur choix actuel : tres bon compromis FP/FN et meilleur F1-score |
| CatBoost | 0.8947 | 0.8673 | 0.8808 | 0.9996 | 0.9781 | 0.8780 | 10 | 13 | `CatBoostClassifier`, `iterations=300`, `learning_rate=0.09`, `depth=6`, `class_weights=[1, 40]`, `threshold=0.4`, `loss_function=Logloss`, `eval_metric=AUC`, petite grille : 24 configurations de modele entrainees, 120 combinaisons modele + threshold evaluees; option plus prudente : FP 6, FN 16 avec `iterations=400`, `learning_rate=0.05`, `depth=8`, `threshold=0.6` | Tres proche de Random Forest, avec peu de FP et peu de FN |
| XGBoost | 0.9222 | 0.8469 | 0.8830 | 0.9996 | 0.9728 | 0.8874 | 7 | 15 | `StandardScaler`, `XGBClassifier`, `n_estimators=200`, `learning_rate=0.2`, `max_depth=5`, `scale_pos_weight=25`, `threshold=0.6`, 72 modeles entraines, 648 combinaisons evaluees | Excellent PR-AUC et tres peu de FP, mais plus de FN que Random Forest et CatBoost |
| LightGBM | 0.9111 | 0.8367 | 0.8723 | 0.9996 | 0.9717 | 0.8584 | 8 | 16 | `LGBMClassifier`, `n_estimators=200`, `learning_rate=0.03`, `max_depth=4`, `num_leaves=15`, `min_child_samples=100`, `scale_pos_weight=1`, `threshold=0.3`, 540 modeles entraines, 4968 combinaisons evaluees | Bon modele, proche de XGBoost, mais avec un peu plus de FN |
| SVM | 0.8367 | 0.8367 | 0.8367 | 0.9994 | 0.9742 | 0.7516 | 16 | 16 | `StandardScaler`, `LinearSVC`, `C=0.001`, `class_weight={0: 1, 1: 40}`, `threshold=0.6`, petite grille : 25 modeles entraines, 175 combinaisons evaluees | Resultat correct, mais moins competitif que les modeles d'arbres |
| KNN | 0.5959 | 0.8878 | 0.7131 | 0.9988 | 0.9484 | 0.8308 | 59 | 11 | `StandardScaler`, `KNeighborsClassifier`, train reduit : toutes les fraudes + 40000 transactions normales, `n_neighbors=9`, `weights=distance`, `threshold=0.2`, petite grille : 30 modeles entraines, 150 combinaisons evaluees | Meilleur FN du tableau, mais beaucoup trop de FP pour passer devant les meilleurs modeles |
| Logistic Regression | 0.7155 | 0.8469 | 0.7757 | 0.9992 | 0.9706 | 0.7449 | 33 | 15 | `StandardScaler`, `C=0.01`, `class_weight={0: 1, 1: 10}`, `threshold=0.6`, `max_iter=3000`, petite grille : 30 modeles entraines, 150 combinaisons evaluees | Bonne baseline interpretable, mais trop de FP et F1-score plus faible |
| Gradient Boosting | 0.8889 | 0.8163 | 0.8511 | 0.9995 | 0.9792 | 0.8540 | 10 | 18 | `StandardScaler`, `HistGradientBoostingClassifier`, `learning_rate=0.05`, `max_iter=100`, `max_leaf_nodes=15`, `min_samples_leaf=20`, `class_weight={0: 1, 1: 5}`, `threshold=0.6`, 144 modeles entraines, 1296 combinaisons evaluees | Peu de FP, mais trop de FN face aux meilleurs modeles |
| Isolation Forest | 0.3509 | 0.4082 | 0.3774 | 0.9977 | 0.9612 | 0.3292 | 74 | 58 | `StandardScaler`, `IsolationForest`, `n_estimators=200`, `max_samples=4096`, `max_features=0.5`, `top_anomaly_rate=0.002`, non supervise, 48 modeles entraines, 288 combinaisons evaluees | Interessant comme detecteur non supervise, mais pas competitif ici |

## Pistes d'exploration (upgrades simples)

- **Empilement (stacking)** : combiner plusieurs modèles de base (ex. `RandomForest`, `XGBoost`, `CatBoost`) et entraîner un méta-modèle (ex. `LogisticRegression`) sur leurs prédictions out-of-fold. Avantage : capte signaux complémentaires, améliore souvent F1/PR-AUC.
- **AutoML** : tester `AutoGluon` ou `H2O AutoML` pour rechercher automatiquement de bonnes combinaisons et préprocessing. Avantage : prototypes rapides, découvre combos inattendus.
- **IsolationForest comme feature** : utiliser le score d'anomalie comme variable d'entrée pour les modèles supervisés. Avantage : permet aux classifieurs de profiter d'un signal non supervisé.
- **Blending / vote pondéré** : agréger probabilités/predictions de plusieurs modèles avec des poids. Avantage : simple à implémenter et souvent robuste.
- **Systèmes en cascade** : filtrer d'abord avec un détecteur d'anomalies, puis classifier les candidats par un modèle supervisé. Avantage : réduit faux positifs et charge de détection.
- **Losss adaptées** : focal loss ou poids de classe (`scale_pos_weight`) pour gérer le déséquilibre. Avantage : améliore détection de la classe rare.
