import altair as alt
import pandas as pd
import streamlit as st

from utils import (
    BINS_BMI, COULEUR_NEUTRE, COULEUR_TRAIN, LABELS_BMI, LIBELLES, barres_horizontales, barres_par_statut,
    charger_donnees, charger_modele, echelle_statut, performances_modeles, preparer_donnees, score_validation_croisee, usd,
)

MODELE_RETENU = "Random Forest (optimisé)"
NOMS_VARIABLES = {**LIBELLES, "smoker_yes": "Fumeur", "sex_male": "Sexe (homme)"}

df = charger_donnees()
donnees = preparer_donnees()
modele, preprocessor = charger_modele()
perf = performances_modeles()

st.title("Choix du modèle")
st.caption(
    "Quatre modèles de régression comparés sur le même jeu de test (20 % des assurés, random_state = 42), "
    "puis optimisation du meilleur, diagnostic du surapprentissage et analyse de ses erreurs."
)

onglet_comparaison, onglet_optimisation, onglet_surapprentissage, onglet_erreurs, onglet_variables = st.tabs(
    ["Comparaison", "Optimisation", "Surapprentissage", "Erreurs", "Variables"]
)

# ---------------------------------------------------------------------------
# Comparaison des quatre modèles
# ---------------------------------------------------------------------------
with onglet_comparaison:
    st.subheader("Quel modèle prédit le mieux les frais médicaux ?")
    st.markdown(
        "- **MAE** : erreur moyenne en USD (plus petit = mieux)\n"
        "- **RMSE** : comme la MAE, mais pénalise davantage les grosses erreurs (plus petit = mieux)\n"
        "- **R²** : part de la variabilité des frais expliquée par le modèle (plus proche de 1 = mieux)"
    )
    tableau = perf[["R2", "MAE", "RMSE"]].sort_values("R2", ascending=False)
    st.dataframe(
        tableau,
        column_config={
            "R2": st.column_config.NumberColumn("R²", format="%.3f"),
            "MAE": st.column_config.NumberColumn("MAE (USD)", format="%.0f"),
            "RMSE": st.column_config.NumberColumn("RMSE (USD)", format="%.0f"),
        },
    )
    st.altair_chart(barres_horizontales(tableau["R2"], "R² sur le jeu de test", format_valeur=".3f"))
    defaut = perf.loc["Random Forest"]
    lineaire = perf.loc["Régression linéaire"]
    st.info(
        f"**Le Random Forest est le meilleur modèle** (R² {defaut['R2']:.3f}, erreur moyenne {usd(defaut['MAE'])}), devant la "
        f"régression linéaire (R² {lineaire['R2']:.3f}). Les frais dépendent des variables de façon non linéaire "
        "(par exemple l'effet du BMI n'existe que chez les fumeurs) : un modèle à base d'arbres le capte, une droite non."
    )

# ---------------------------------------------------------------------------
# Optimisation du Random Forest
# ---------------------------------------------------------------------------
with onglet_optimisation:
    st.subheader("L'optimisation des hyperparamètres améliore-t-elle le Random Forest ?")
    avant = perf.loc["Random Forest"]
    apres = perf.loc[MODELE_RETENU]
    with st.container(horizontal=True):
        st.metric("R² (test)", f"{apres['R2']:.3f}", f"{apres['R2'] - avant['R2']:+.3f} vs défaut", border=True)
        st.metric("MAE (test)", usd(apres["MAE"]), f"{apres['MAE'] - avant['MAE']:+,.0f} USD vs défaut".replace(",", " "),
                  delta_color="inverse", border=True)
        st.metric("RMSE (test)", usd(apres["RMSE"]), f"{apres['RMSE'] - avant['RMSE']:+,.0f} USD vs défaut".replace(",", " "),
                  delta_color="inverse", border=True)
    parametres = modele.get_params()
    st.markdown(
        "**Hyperparamètres retenus par la recherche sur grille (GridSearchCV)** : "
        f"`n_estimators = {parametres['n_estimators']}`, `max_depth = {parametres['max_depth']}`, "
        f"`min_samples_leaf = {parametres['min_samples_leaf']}`."
    )
    st.info(
        f"Limiter la profondeur des arbres et imposer un minimum d'assurés par feuille rend le modèle moins sensible au bruit : "
        f"le R² passe de {avant['R2']:.3f} à {apres['R2']:.3f} et l'erreur moyenne de {usd(avant['MAE'])} à {usd(apres['MAE'])}. "
        "Un Gradient Boosting testé dans le notebook atteint un score équivalent (R² ≈ 0.90) : nous gardons le Random Forest, "
        "plus simple à expliquer."
    )

# ---------------------------------------------------------------------------
# Surapprentissage : entraînement contre test
# ---------------------------------------------------------------------------
with onglet_surapprentissage:
    st.subheader("Le modèle apprend-il par cœur ou généralise-t-il ?")
    st.caption("Un modèle qui surapprend a un excellent score sur les données qu'il a vues (train) et un score bien plus faible sur des données neuves (test).")
    longue = (
        perf[["R2 train", "R2"]]
        .rename(columns={"R2 train": "Entraînement", "R2": "Test"})
        .rename_axis("modele")
        .reset_index()
        .melt(id_vars="modele", var_name="jeu", value_name="r2")
    )
    base = alt.Chart(longue).encode(
        y=alt.Y("modele:N", sort=list(perf.index), title=None, axis=alt.Axis(labelLimit=260)),
        yOffset=alt.YOffset("jeu:N", sort=["Entraînement", "Test"]),
        x=alt.X("r2:Q", title="R²", scale=alt.Scale(domain=[0, 1.05])),
        color=alt.Color(
            "jeu:N", title="Jeu", sort=["Entraînement", "Test"],
            scale=alt.Scale(domain=["Entraînement", "Test"], range=[COULEUR_TRAIN, COULEUR_NEUTRE]),
        ),
        tooltip=[alt.Tooltip("modele:N", title="Modèle"), alt.Tooltip("jeu:N", title="Jeu"), alt.Tooltip("r2:Q", title="R²", format=".3f")],
    )
    etiquettes = base.mark_text(align="left", dx=4).encode(text=alt.Text("r2:Q", format=".3f"))
    st.altair_chart((base.mark_bar() + etiquettes).properties(height=380))
    ecart = (perf["R2 train"] - perf["R2"]).rename("écart")
    valide = score_validation_croisee()
    st.info(
        f"Le Random Forest par défaut passe de {perf.loc['Random Forest', 'R2 train']:.3f} (train) à {perf.loc['Random Forest', 'R2']:.3f} "
        f"(test) : il apprend un peu trop par cœur (écart {ecart['Random Forest']:.3f}), comme l'arbre de décision "
        f"({ecart['Arbre de décision']:.3f}). **Le modèle optimisé** limite ce phénomène : écart de seulement {ecart[MODELE_RETENU]:.3f} "
        f"(R² {perf.loc[MODELE_RETENU, 'R2 train']:.3f} en train contre {perf.loc[MODELE_RETENU, 'R2']:.3f} en test), "
        f"et un R² moyen de {valide:.3f} en validation croisée à 5 découpages sur le train. Il généralise correctement."
    )

# ---------------------------------------------------------------------------
# Analyse des erreurs du modèle retenu
# ---------------------------------------------------------------------------
with onglet_erreurs:
    st.subheader("Où le modèle retenu se trompe-t-il ?")
    test = donnees["X_test"].copy()
    test["reel"] = donnees["y_test"]
    test["predit"] = modele.predict(donnees["X_test_p"])
    test["residu"] = test["reel"] - test["predit"]
    test["statut"] = df.loc[test.index, "statut"]

    with st.container(horizontal=True):
        st.metric("Erreur moyenne signée", usd(test["residu"].mean()), border=True)
        st.metric("Erreur absolue moyenne (MAE)", usd(test["residu"].abs().mean()), border=True)
        st.metric("Plus forte sous-estimation", usd(test["residu"].max()), border=True)
    st.caption("Une erreur négative signifie que le modèle surestime en moyenne, une erreur positive qu'il sous-estime.")

    st.markdown("#### Frais réels contre frais prédits")
    maximum = float(max(test["reel"].max(), test["predit"].max()))
    nuage = (
        alt.Chart(test)
        .mark_circle(size=50, opacity=0.7)
        .encode(
            x=alt.X("reel:Q", title="Frais réels (USD)"),
            y=alt.Y("predit:Q", title="Frais prédits (USD)"),
            color=alt.Color("statut:N", scale=echelle_statut(), legend=alt.Legend(title="Statut")),
            tooltip=[
                alt.Tooltip("statut:N", title="Statut"), alt.Tooltip("age:Q", title="Âge"), alt.Tooltip("bmi:Q", title="BMI"),
                alt.Tooltip("reel:Q", title="Réel", format=",.0f"), alt.Tooltip("predit:Q", title="Prédit", format=",.0f"),
            ],
        )
    )
    diagonale = (
        alt.Chart(pd.DataFrame({"x": [0, maximum], "y": [0, maximum]}))
        .mark_line(strokeDash=[5, 5], color="#0b0b0b")
        .encode(x="x:Q", y="y:Q")
    )
    st.altair_chart((nuage + diagonale).properties(height=380).interactive())
    st.caption("Plus un point est proche de la ligne pointillée (prédiction parfaite), meilleure est la prédiction.")

    st.markdown("#### Distribution des erreurs")
    histogramme = (
        alt.Chart(test)
        .mark_bar(color=COULEUR_NEUTRE)
        .encode(
            x=alt.X("residu:Q", bin=alt.Bin(maxbins=30), title="Erreur = frais réels − frais prédits (USD)"),
            y=alt.Y("count():Q", title="Nombre d'assurés"),
        )
        .properties(height=300)
    )
    st.altair_chart(histogramme)

    st.markdown("#### Erreur moyenne selon le profil")
    categorie = pd.cut(test["bmi"], bins=BINS_BMI, labels=LABELS_BMI).rename("categorie")
    erreur_profil = (
        test.assign(erreur=test["residu"].abs())
        .groupby([categorie, "statut"], observed=True)["erreur"]
        .mean()
        .reset_index(name="valeur")
    )
    st.altair_chart(
        barres_par_statut(erreur_profil, "categorie", "valeur", ordre=LABELS_BMI, titre_x="Catégorie de BMI",
                          titre_y="Erreur absolue moyenne (USD)")
    )
    plus_grosses = test.assign(erreur=test["residu"].abs()).nlargest(10, "erreur")
    sous_estimees = int((plus_grosses["residu"] > 0).sum())
    st.info(
        f"L'erreur est du même ordre chez les fumeurs ({usd(test.loc[test['statut'] == 'Fumeur', 'residu'].abs().mean())}) et "
        f"chez les non-fumeurs ({usd(test.loc[test['statut'] == 'Non-fumeur', 'residu'].abs().mean())}). Parmi les 10 plus "
        f"grosses erreurs, {sous_estimees} sont des **sous-estimations** : le modèle a du mal avec les coûts extrêmes "
        "(accidents, maladies graves) que rien dans les six variables ne permet d'anticiper."
    )

    st.markdown("#### Les 10 plus grosses erreurs")
    tableau_erreurs = plus_grosses[["age", "bmi", "children", "statut", "reel", "predit", "residu"]]
    st.dataframe(
        tableau_erreurs,
        column_config={
            "age": st.column_config.NumberColumn("Âge", format="%.0f"),
            "bmi": st.column_config.NumberColumn("BMI", format="%.1f"),
            "children": st.column_config.NumberColumn("Enfants", format="%.0f"),
            "statut": "Statut",
            "reel": st.column_config.NumberColumn("Frais réels (USD)", format="%.0f"),
            "predit": st.column_config.NumberColumn("Frais prédits (USD)", format="%.0f"),
            "residu": st.column_config.NumberColumn("Erreur (USD)", format="%.0f"),
        },
        hide_index=True,
    )

# ---------------------------------------------------------------------------
# Importance des variables
# ---------------------------------------------------------------------------
with onglet_variables:
    st.subheader("Quelles variables le modèle utilise-t-il pour prédire ?")
    noms = [nom.split("__", 1)[-1] for nom in preprocessor.get_feature_names_out()]
    importances = pd.Series(modele.feature_importances_, index=[NOMS_VARIABLES.get(nom, nom) for nom in noms])
    importances = importances.sort_values(ascending=False)
    st.altair_chart(barres_horizontales(importances, "Importance (part de la décision du modèle)", format_valeur=".1%", hauteur=300))
    st.info(
        f"Le statut fumeur pèse à lui seul **{importances['Fumeur']:.0%}** des décisions du modèle, suivi du BMI "
        f"({importances['BMI']:.0%}) et de l'âge ({importances['Âge']:.0%}). Le sexe, le nombre d'enfants et la région "
        f"se partagent les {importances.drop(['Fumeur', 'BMI', 'Âge']).sum():.0%} restants : c'est cohérent avec l'analyse "
        "exploratoire, qui ne trouvait aucun effet démographique marqué."
    )
