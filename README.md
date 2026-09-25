# Aegis Health Coverage : tarification de l'assurance santé

Projet NexaData Consulting (Machine Learning, régression) réalisé par **Suz, David et Loïc**.

Aegis veut automatiser l'estimation des frais médicaux annuels de ses assurés, aujourd'hui faite à la main (erreurs d'appréciation, marges volatiles). Ce dépôt contient l'analyse, le modèle de régression et une application Streamlit.

## Contenu

| Élément | Rôle |
|---|---|
| `insurance_health_prediction.ipynb` | Notebook : exploration, 3 lots de KPI, comparaison de 4 modèles, optimisation, diagnostic, sauvegarde du modèle |
| `streamlit_app.py` | Point d'entrée de l'application (navigation) |
| `app_pages/` | Pages : Accueil, Analyse, Choix du modèle, Simulateur, Recommandation |
| `utils.py` | Chargement des données et du modèle, calculs et graphiques partagés |
| `data/insurance-data.csv` | Données (1 338 lignes, 1 337 après suppression d'un doublon) |
| `model_rf.pkl`, `preprocessor.pkl` | Random Forest optimisé et prétraitement produits par le notebook |

## Résultats

| Modèle (jeu de test) | R² | MAE (USD) | RMSE (USD) |
|---|---|---|---|
| Random Forest optimisé | 0,898 | 2 459 | 4 329 |
| Random Forest (défaut) | 0,879 | 2 665 | 4 715 |
| Régression linéaire | 0,807 | 4 177 | 5 957 |
| Arbre de décision | 0,788 | 2 917 | 6 238 |
| KNN | 0,646 | 4 495 | 8 065 |

Le statut fumeur explique à lui seul environ 66 % des décisions du modèle, devant le BMI (19 %) et l'âge (13 %).

## Lancer l'application

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Les fichiers `model_rf.pkl` et `preprocessor.pkl` ont été créés avec scikit-learn 1.7.2 : gardez cette version pour les charger sans erreur.

## Équipe

| Membre | Rôle | Lot d'analyse |
|---|---|---|
| Loïc | Data Analyst | Lot 1 · Tabagisme |
| Suz | Data Scientist | Lot 2 · BMI et âge |
| David | Data Analyst | Lot 3 · Région et famille |
