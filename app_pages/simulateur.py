import altair as alt
import pandas as pd
import streamlit as st

from utils import (
    FEATURES, ORDRE_STATUT, REGIONS, charger_donnees, charger_modele, echelle_statut, evaluer_modele, usd,
)

df = charger_donnees()
modele, preprocessor = charger_modele()
mae = evaluer_modele()["MAE"]
frais_moyens = df["expenses"].mean()


def predire(profil):
    """Frais annuels prédits pour un profil (dictionnaire avec les six variables du modèle)."""
    ligne = pd.DataFrame([profil])[FEATURES]
    return float(modele.predict(preprocessor.transform(ligne))[0])


st.title("Simulateur de prime")
st.caption("Décrivez un assuré : le Random Forest optimisé estime ses frais médicaux annuels, base d'une prime d'assurance.")

colonne_saisie, colonne_resultat = st.columns([1, 1.4], gap="large")

with colonne_saisie:
    with st.container(border=True):
        st.markdown("#### :material/person: Profil de l'assuré")
        age = st.slider("Âge", 18, 64, 35)
        sexe = st.radio("Sexe", ["Femme", "Homme"], horizontal=True)
        bmi = st.number_input("BMI (poids en kg ÷ taille² en m²)", min_value=15.0, max_value=55.0, value=28.0, step=0.5)
        enfants = st.slider("Enfants à charge", 0, 5, 0)
        fumeur = st.radio("Statut", ["Non-fumeur", "Fumeur"], horizontal=True)
        region = st.selectbox("Région", list(REGIONS), format_func=REGIONS.get, index=2)

profil = {
    "age": age,
    "sex": "male" if sexe == "Homme" else "female",
    "bmi": bmi,
    "children": enfants,
    "smoker": "yes" if fumeur == "Fumeur" else "no",
    "region": region,
}
estimation = predire(profil)
sans_tabac = predire({**profil, "smoker": "no"})
avec_tabac = predire({**profil, "smoker": "yes"})

with colonne_resultat:
    with st.container(border=True):
        st.markdown("#### :material/payments: Estimation des frais annuels")
        with st.container(horizontal=True):
            st.metric("Frais annuels estimés", usd(estimation), border=True)
            st.metric("Écart avec la moyenne du portefeuille", f"{estimation / frais_moyens - 1:+.0%}", border=True)
        st.caption(
            f"La moyenne du portefeuille est de {usd(frais_moyens)} par assuré. Ordre de grandeur de l'estimation : "
            f"{usd(max(estimation - mae, 0))} à {usd(estimation + mae)} (± l'erreur moyenne du modèle, {usd(mae)})."
        )

        if profil["smoker"] == "yes" and bmi >= 30:
            st.warning("Profil « fumeur + BMI élevé » : c'est la catégorie la plus coûteuse du portefeuille (voir l'analyse BMI et âge).")
        elif profil["smoker"] == "yes":
            st.warning("Le statut fumeur est le premier facteur de coût : il multiplie les frais estimés.")
        else:
            st.success("Profil non-fumeur : aucune surcharge liée au tabac dans l'estimation.")

    with st.container(border=True):
        st.markdown("#### :material/compare_arrows: Le même profil, avec et sans tabac")
        comparaison = pd.DataFrame({"statut": ORDRE_STATUT, "frais": [sans_tabac, avec_tabac]})
        base = alt.Chart(comparaison).encode(
            x=alt.X("statut:N", sort=ORDRE_STATUT, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("frais:Q", title="Frais annuels estimés (USD)"),
            color=alt.Color("statut:N", scale=echelle_statut(), legend=None),
            tooltip=[alt.Tooltip("statut:N", title="Statut"), alt.Tooltip("frais:Q", title="Frais estimés", format=",.0f")],
        )
        etiquettes = base.mark_text(dy=-8, fontWeight="bold").encode(text=alt.Text("frais:Q", format=",.0f"))
        st.altair_chart((base.mark_bar() + etiquettes).properties(height=280))
        st.info(
            f"À caractéristiques égales, passer de non-fumeur à fumeur fait varier l'estimation de "
            f"**{usd(avec_tabac - sans_tabac)}** par an ({avec_tabac / sans_tabac:.1f} fois plus)."
        )

st.caption(
    "Estimation fournie à titre indicatif par un modèle entraîné sur environ 1 300 assurés : c'est une aide à la décision, "
    "pas un prix contractuel."
)
