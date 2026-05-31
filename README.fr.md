# Credit Card Fraud Detection

> English version: [README.md](README.md)

Projet de test de plusieurs modèles de machine learning sur le dataset Kaggle
`mlg-ulb/creditcardfraud`.

Le dataset est très déséquilibré :

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

Le fichier `creditcard.csv` est téléchargé localement à côté de `data-import.py`.

## Tests effectués

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
rapide. KNN et SVM sont plutôt à tester plus tard,
car ils peuvent devenir lents sur un dataset de cette taille.

## Fichiers

- `data-import.py` : téléchargement du dataset
- `utils/model_utils.py` : chargement des données, split train/test et affichage des résultats
- `utils/feature_utils.py` : création de features expérimentales
- `models/logistic-regression.py`
- `models/random-forest.py`
- `models/gradient-boosting.py` : utilise `HistGradientBoostingClassifier`, une version plus rapide du gradient boosting sklearn
- `models/xgboost-model.py`
- `models/lightgbm-model.py`
- `models/catboost-model.py`
- `models/knn.py`
- `models/svm.py`
- `models/isolation-forest.py`
- `experiments/isolation-feature-test.py` : test du score Isolation Forest comme nouvelle feature
- `experiments/blending-test.py` : test d'une moyenne pondérée entre plusieurs modèles
- `experiments/blending-knn-test.py` : test d'une moyenne pondérée avec un petit poids KNN
- `experiments/false-negative-overlap.py` : vérification des fraudes ratées par chaque modèle

## Métriques comparées

Comme les fraudes sont rares, l'accuracy seule n'est pas suffisante.

Suivi pour chaque modèle :

| Métrique | Description | Valeur haute | Valeur basse |
| --- | --- | --- | --- |
| Précision | Parmi les transactions prédites comme fraude, combien sont vraiment des fraudes. | Peu de fausses alertes. | Beaucoup de fausses alertes. |
| Recall | Parmi les vraies fraudes, combien sont trouvées par le modèle. | Peu de fraudes ratées. | Beaucoup de fraudes ratées. |
| F1-score | Équilibre entre précision et recall. | Bon compromis. | Précision ou recall trop faible. |
| Accuracy | Part totale de bonnes prédictions. | Beaucoup de bonnes prédictions. | Beaucoup d'erreurs, mais peut être trompeuse si les classes sont déséquilibrées. |
| ROC-AUC | Capacité du modèle à donner un score plus haut aux fraudes qu'aux transactions normales. | Bon classement global. | Mauvais classement. |
| PR-AUC | Qualité du compromis précision/recall. | Bon résultat sur la classe rare. | Mauvais compromis pour détecter les fraudes. |
| FP | False positives : transactions normales prédites comme fraude. | Beaucoup de fausses alertes. | Peu de fausses alertes. |
| FN | False negatives : fraudes prédites comme normales. | Beaucoup de fraudes ratées. | Peu de fraudes ratées. |
| Confusion matrix | Détail des bonnes et mauvaises prédictions. | Permet de voir les bonnes prédictions. | Permet de voir les fausses alertes et les fraudes ratées. |

## Résultats

Le tableau est trié selon un compromis FN/FP : on privilégie les faux négatifs bas,
mais un petit gain en FN ne justifie pas une forte hausse des faux positifs.

| Modèle | Précision | Recall | F1-score | Accuracy | ROC-AUC | PR-AUC | FP | FN | Notes | Conclusion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Random Forest | 0.9247 | 0.8776 | 0.9005 | 0.9997 | 0.9581 | 0.8678 | 7 | 12 | `StandardScaler`, `n_estimators=90`, `class_weight={0: 1, 1: 15}`, `threshold=0.3`, batch partiel : 36 modèles entraînés, 540 combinaisons évaluées ; option plus agressive : FP 38, FN 10 avec `n_estimators=60`, `class_weight=balanced`, `max_depth=24`, `min_samples_leaf=2`, `max_features=sqrt`, `threshold=0.1` | Meilleur choix actuel : très bon compromis FP/FN et meilleur F1-score |
| CatBoost | 0.8947 | 0.8673 | 0.8808 | 0.9996 | 0.9781 | 0.8780 | 10 | 13 | `CatBoostClassifier`, `iterations=300`, `learning_rate=0.09`, `depth=6`, `class_weights=[1, 40]`, `threshold=0.4`, `loss_function=Logloss`, `eval_metric=AUC`, petite grille : 24 configurations de modèle entraînées, 120 combinaisons modèle + threshold évaluées ; option plus prudente : FP 6, FN 16 avec `iterations=400`, `learning_rate=0.05`, `depth=8`, `threshold=0.6` | Très proche de Random Forest, avec peu de FP et peu de FN |
| XGBoost | 0.9222 | 0.8469 | 0.8830 | 0.9996 | 0.9728 | 0.8874 | 7 | 15 | `StandardScaler`, `XGBClassifier`, `n_estimators=200`, `learning_rate=0.2`, `max_depth=5`, `scale_pos_weight=25`, `threshold=0.6`, 72 modèles entraînés, 648 combinaisons évaluées | Excellent PR-AUC et très peu de FP, mais plus de FN que Random Forest et CatBoost |
| LightGBM | 0.9111 | 0.8367 | 0.8723 | 0.9996 | 0.9717 | 0.8584 | 8 | 16 | `LGBMClassifier`, `n_estimators=200`, `learning_rate=0.03`, `max_depth=4`, `num_leaves=15`, `min_child_samples=100`, `scale_pos_weight=1`, `threshold=0.3`, 540 modèles entraînés, 4968 combinaisons évaluées | Bon modèle, proche de XGBoost, mais avec un peu plus de FN |
| SVM | 0.8367 | 0.8367 | 0.8367 | 0.9994 | 0.9742 | 0.7516 | 16 | 16 | `StandardScaler`, `LinearSVC`, `C=0.001`, `class_weight={0: 1, 1: 40}`, `threshold=0.6`, petite grille : 25 modèles entraînés, 175 combinaisons évaluées | Résultat correct, mais moins compétitif que les modèles d'arbres |
| KNN | 0.5959 | 0.8878 | 0.7131 | 0.9988 | 0.9484 | 0.8308 | 59 | 11 | `StandardScaler`, `KNeighborsClassifier`, train réduit : toutes les fraudes + 40000 transactions normales, `n_neighbors=9`, `weights=distance`, `threshold=0.2`, petite grille : 30 modèles entraînés, 150 combinaisons évaluées | Meilleur FN du tableau, mais beaucoup trop de FP pour passer devant les meilleurs modèles |
| Logistic Regression | 0.7155 | 0.8469 | 0.7757 | 0.9992 | 0.9706 | 0.7449 | 33 | 15 | `StandardScaler`, `C=0.01`, `class_weight={0: 1, 1: 10}`, `threshold=0.6`, `max_iter=3000`, petite grille : 30 modèles entraînés, 150 combinaisons évaluées | Bonne baseline interprétable, mais trop de FP et F1-score plus faible |
| Gradient Boosting | 0.8889 | 0.8163 | 0.8511 | 0.9995 | 0.9792 | 0.8540 | 10 | 18 | `StandardScaler`, `HistGradientBoostingClassifier`, `learning_rate=0.05`, `max_iter=100`, `max_leaf_nodes=15`, `min_samples_leaf=20`, `class_weight={0: 1, 1: 5}`, `threshold=0.6`, 144 modèles entraînés, 1296 combinaisons évaluées | Peu de FP, mais trop de FN face aux meilleurs modèles |
| Isolation Forest | 0.3509 | 0.4082 | 0.3774 | 0.9977 | 0.9612 | 0.3292 | 74 | 58 | `StandardScaler`, `IsolationForest`, `n_estimators=200`, `max_samples=4096`, `max_features=0.5`, `top_anomaly_rate=0.002`, non supervisé, 48 modèles entraînés, 288 combinaisons évaluées | Intéressant comme détecteur non supervisé, mais pas compétitif ici |

## Expériences

### Isolation Forest comme feature

Test dans `experiments/isolation-feature-test.py`.

Objectif : ajouter le score d'anomalie de `IsolationForest` comme nouvelle colonne
pour aider les modèles supervisés.

| Modèle | Résultat FN/FP retenu avec `isolation_score` | Comparaison au modèle principal |
| --- | --- | --- |
| Random Forest | FP 14, FN 12, F1 0.8687 | FN identique, mais FP plus élevé |
| CatBoost | FP 10, FN 16, F1 0.8632 | Moins bon : plus de FN |
| XGBoost | FP 13, FN 15, F1 0.8557 | FN identique, mais FP plus élevé |
| LightGBM | FP 16, FN 18, F1 0.8247 | Moins bon : plus de FP et plus de FN |

Conclusion : la feature `isolation_score` n'améliore pas les meilleurs modèles
avec les paramètres actuels. Elle reste intéressante à garder comme idée pour
un futur stacking, mais elle n'est pas ajoutée au tableau principal.

### Blending simple

Test dans `experiments/blending-test.py`.

Objectif : mélanger les scores de `Random Forest`, `CatBoost` et `XGBoost`
avec une moyenne pondérée.

| Résultat | FP | FN | F1-score | Poids | Comparaison |
| --- | --- | --- | --- | --- | --- |
| Meilleur F1 | 4 | 14 | 0.9032 | RF 0.90, CatBoost 0.05, XGBoost 0.05, threshold 0.43 | F1 légèrement meilleur, mais plus de FN |
| Meilleur FN bas avec FP <= 50 | 8 | 12 | 0.8958 | RF 0.80, CatBoost 0.15, XGBoost 0.05, threshold 0.27 | FN identique à Random Forest, mais FP plus élevé |

Grille : 171 combinaisons de poids, 46 thresholds, 7866 combinaisons évaluées.

Conclusion : le blending simple est intéressant, mais il ne remplace pas encore
Random Forest comme meilleur choix principal.

### Blending avec KNN léger

Test dans `experiments/blending-knn-test.py`.

Objectif : ajouter un petit poids KNN au blending pour voir s'il réduit les FN
sans faire exploser les FP.

| Résultat | FP | FN | F1-score | Poids | Comparaison |
| --- | --- | --- | --- | --- | --- |
| Meilleur F1 | 5 | 14 | 0.8984 | RF 0.80, CatBoost 0.05, XGBoost 0.10, KNN 0.05, threshold 0.45 | Bon F1-score, mais plus de FN que Random Forest |
| Meilleur FN bas avec FP <= 50 | 37 | 11 | 0.7838 | RF 0.20, CatBoost 0.70, XGBoost 0.05, KNN 0.05, threshold 0.10 | FN plus bas, mais FP beaucoup plus élevé |

Grille : 605 combinaisons de poids, 56 thresholds, 33880 combinaisons évaluées.

Conclusion : le KNN apporte bien un signal différent pour réduire les FN,
mais le prix en FP est trop élevé. Le résultat ne remplace pas Random Forest
comme meilleur compromis actuel.

### Overlap des faux négatifs

Test dans `experiments/false-negative-overlap.py`.

Objectif : vérifier si les modèles ratent les mêmes fraudes.

Modèles testés : Random Forest, CatBoost, XGBoost, LightGBM, Gradient Boosting,
Logistic Regression et SVM.

| Analyse | Résultat |
| --- | --- |
| Fraudes dans le test set | 98 |
| Fraudes ratées par au moins un modèle | 22 |
| Fraudes ratées par tous les modèles testés | 12 |

Fraudes ratées par tous les modèles testés :
`623`, `10497`, `50537`, `68067`, `72757`, `96341`, `119714`, `157585`,
`157918`, `182992`, `219025`, `245347`.

Conclusion : cela explique pourquoi le blending n'améliore pas vraiment le
meilleur modèle : les modèles se trompent sur les mêmes cas difficiles. La
priorité n'est donc plus de mélanger plus de modèles, mais de comprendre ces
12 fraudes spéciales.

## Suite du projet

Les prochains tests doivent partir du constat principal : les 12 fraudes ratées
par Random Forest sont aussi ratées par les autres modèles testés.

1. **Analyser les 12 fraudes ratées**  
   Regarder leurs valeurs (`Time`, `Amount`, `V1` à `V28`) pour comprendre si
   elles ressemblent plus à des transactions normales qu'aux autres fraudes.

2. **Comparer fraudes ratées vs fraudes détectées**  
   Chercher les colonnes où les 12 cas difficiles sont différents des fraudes
   bien détectées.

3. **Afficher les scores des modèles sur ces 12 cas**  
   Vérifier si les modèles sont totalement confiants que ce sont des transactions
   normales, ou si les scores sont juste un peu sous le threshold.

4. **Tester des features ciblées**  
   Ajouter seulement des features qui viennent de l'analyse des 12 cas, par
   exemple autour de `Amount`, `Time` ou de la distance aux autres fraudes.

5. **Valider sur plusieurs splits**  
   Refaire le test avec plusieurs découpages train/test pour vérifier que le
   problème ne vient pas seulement du split actuel.
