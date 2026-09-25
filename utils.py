from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

RACINE = Path(__file__).parent
RANDOM_STATE = 42
FEATURES = ["age", "sex", "bmi", "children", "smoker", "region"]

STATUT = {"no": "Non-fumeur", "yes": "Fumeur"}
ORDRE_STATUT = ["Non-fumeur", "Fumeur"]
COULEURS = {"Non-fumeur": "#2a78d6", "Fumeur": "#eb6834"}
COULEUR_NEUTRE = "#4a3aa7"
COULEUR_TRAIN = "#898781"
REGIONS = {"northeast": "Nord-Est", "northwest": "Nord-Ouest", "southeast": "Sud-Est", "southwest": "Sud-Ouest"}
LIBELLES = {
    "age": "Âge", "bmi": "BMI", "children": "Enfants", "expenses": "Frais", "sex": "Sexe (homme)",
    "smoker": "Fumeur", "region_northwest": "Région Nord-Ouest", "region_southeast": "Région Sud-Est",
    "region_southwest": "Région Sud-Ouest",
}
BINS_BMI = [0, 25, 30, 100]
LABELS_BMI = ["Normal (<25)", "Surpoids (25-30)", "Obésité (≥30)"]


def usd(valeur):
    """Montant à la française : 13 279 USD."""
    return f"{valeur:,.0f}".replace(",", " ") + " USD"


# ---------------------------------------------------------------------------
# Données et modèle
# ---------------------------------------------------------------------------
@st.cache_data
def charger_donnees():
    df = pd.read_csv(RACINE / "data" / "insurance-data.csv")
    df = df.drop_duplicates().reset_index(drop=True)
    df["statut"] = df["smoker"].map(STATUT)
    return df


@st.cache_resource
def charger_modele():
    """Random Forest optimisé et prétraitement enregistrés par le notebook."""
    modele = joblib.load(RACINE / "model_rf.pkl")
    preprocessor = joblib.load(RACINE / "preprocessor.pkl")
    return modele, preprocessor


@st.cache_data
def preparer_donnees():
    """Même découpage train/test que dans le notebook (80/20, random_state=42), prétraitement du notebook."""
    df = charger_donnees()
    _, preprocessor = charger_modele()
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df["expenses"], test_size=0.2, random_state=RANDOM_STATE
    )
    return {
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "X_train_p": preprocessor.transform(X_train), "X_test_p": preprocessor.transform(X_test),
    }


def evaluer(y_reel, y_predit):
    """MAE, RMSE et R² d'une prédiction."""
    mse = metrics.mean_squared_error(y_reel, y_predit)
    return {
        "MAE": metrics.mean_absolute_error(y_reel, y_predit),
        "RMSE": mse ** 0.5,
        "R2": metrics.r2_score(y_reel, y_predit),
    }


@st.cache_data
def evaluer_modele():
    """Performances du modèle retenu sur le jeu de test du notebook."""
    donnees = preparer_donnees()
    modele, _ = charger_modele()
    return evaluer(donnees["y_test"], modele.predict(donnees["X_test_p"]))


@st.cache_resource
def entrainer_modeles():
    """Les quatre modèles du notebook (paramètres par défaut) et le Random Forest optimisé enregistré."""
    donnees = preparer_donnees()
    modeles = {
        "Régression linéaire": LinearRegression(),
        "KNN": KNeighborsRegressor(),
        "Arbre de décision": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(random_state=RANDOM_STATE),
    }
    for modele in modeles.values():
        modele.fit(donnees["X_train_p"], donnees["y_train"])
    modeles["Random Forest (optimisé)"] = charger_modele()[0]
    return modeles


@st.cache_data
def performances_modeles():
    """Performances de chaque modèle sur le test (MAE, RMSE, R2) et sur le train (MAE train, R2 train)."""
    donnees = preparer_donnees()
    lignes = {}
    for nom, modele in entrainer_modeles().items():
        test = evaluer(donnees["y_test"], modele.predict(donnees["X_test_p"]))
        train = evaluer(donnees["y_train"], modele.predict(donnees["X_train_p"]))
        lignes[nom] = {**test, "R2 train": train["R2"], "MAE train": train["MAE"]}
    return pd.DataFrame(lignes).T


@st.cache_data
def score_validation_croisee():
    """R² moyen en validation croisée (5 découpages) du modèle retenu, sur le train uniquement."""
    donnees = preparer_donnees()
    modele, _ = charger_modele()
    return float(cross_val_score(modele, donnees["X_train_p"], donnees["y_train"], cv=5, scoring="r2").mean())


@st.cache_data
def indicateurs():
    """Chiffres clés de l'exploration, réutilisés dans les recommandations."""
    df = charger_donnees()
    moyennes = df.groupby("statut")["expenses"].mean()
    categorie = pd.cut(df["bmi"], bins=BINS_BMI, labels=LABELS_BMI)
    par_bmi = df.groupby([categorie, "statut"], observed=True)["expenses"].mean().unstack()
    return {
        "ratio": moyennes["Fumeur"] / moyennes["Non-fumeur"],
        "ecart": moyennes["Fumeur"] - moyennes["Non-fumeur"],
        "part_depenses_fumeurs": df.loc[df["smoker"] == "yes", "expenses"].sum() / df["expenses"].sum(),
        "saut_fumeurs": par_bmi.loc[LABELS_BMI[2], "Fumeur"] / par_bmi.loc[LABELS_BMI[1], "Fumeur"] - 1,
        "hausse_non_fumeurs": par_bmi.loc[LABELS_BMI[2], "Non-fumeur"] / par_bmi.loc[LABELS_BMI[0], "Non-fumeur"] - 1,
        "ecart_regions": df.groupby("region")["expenses"].mean().agg(lambda s: s.max() - s.min()),
    }


# ---------------------------------------------------------------------------
# Graphiques
# ---------------------------------------------------------------------------
def echelle_statut():
    """Bleu pour les non-fumeurs, orange pour les fumeurs, partout dans l'application."""
    return alt.Scale(domain=ORDRE_STATUT, range=[COULEURS[statut] for statut in ORDRE_STATUT])


def barres_par_statut(donnees, categorie, valeur, ordre=None, titre_x="", titre_y="Frais moyens (USD)"):
    """Barres côte à côte non-fumeur / fumeur ; `donnees` a les colonnes categorie, statut et valeur."""
    base = alt.Chart(donnees).encode(
        x=alt.X(f"{categorie}:N", sort=ordre, title=titre_x, axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("statut:N", sort=ORDRE_STATUT),
        y=alt.Y(f"{valeur}:Q", title=titre_y),
        color=alt.Color("statut:N", scale=echelle_statut(), legend=alt.Legend(title="Statut")),
        tooltip=[
            alt.Tooltip(f"{categorie}:N", title="Catégorie"),
            alt.Tooltip("statut:N", title="Statut"),
            alt.Tooltip(f"{valeur}:Q", title="Valeur", format=",.0f"),
        ],
    )
    barres = base.mark_bar()
    etiquettes = base.mark_text(dy=-8, fontWeight="bold").encode(text=alt.Text(f"{valeur}:Q", format=",.0f"))
    return (barres + etiquettes).properties(height=350)


def barres_horizontales(serie, titre_x, format_valeur=",.0f", couleur=COULEUR_NEUTRE, hauteur=260):
    """Barres horizontales triées, avec la valeur affichée au bout ; `serie` a pour index les catégories."""
    donnees = serie.rename("valeur").rename_axis("categorie").reset_index()
    base = alt.Chart(donnees).encode(
        y=alt.Y("categorie:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260)),
        x=alt.X("valeur:Q", title=titre_x),
        tooltip=[alt.Tooltip("categorie:N", title="Catégorie"), alt.Tooltip("valeur:Q", title="Valeur", format=format_valeur)],
    )
    barres = base.mark_bar(color=couleur)
    etiquettes = base.mark_text(align="left", dx=4, fontWeight="bold").encode(text=alt.Text("valeur:Q", format=format_valeur))
    return (barres + etiquettes).properties(height=hauteur)
