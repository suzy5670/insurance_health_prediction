import streamlit as st

from utils import charger_donnees, evaluer_modele, usd

df = charger_donnees()
performances = evaluer_modele()
nb_assures = f"{len(df):,}".replace(",", " ")

st.title(":material/health_and_safety: Aegis Health Coverage")
st.caption("Tarification de l'assurance santé : de l'analyse des frais médicaux à la simulation d'une prime")

with st.container(horizontal=True):
    st.metric("Assurés analysés", nb_assures, border=True)
    st.metric("Frais moyens par assuré", usd(df["expenses"].mean()), border=True)
    st.metric("Part de fumeurs", f"{(df['smoker'] == 'yes').mean():.1%}", border=True)
    st.metric("R² du modèle (test)", f"{performances['R2']:.3f}", border=True)

st.subheader("Le projet")
col1, col2, col3 = st.columns(3)
with col1:
    with st.container(border=True):
        st.markdown("#### :material/help: Le problème")
        st.write(
            "L'estimation manuelle des contrats d'assurance génère des erreurs d'appréciation et une forte "
            "volatilité des marges. Aegis veut automatiser sa tarification."
        )
with col2:
    with st.container(border=True):
        st.markdown("#### :material/database: Les données")
        st.write(
            f"{nb_assures} assurés décrits par leur âge, sexe, BMI, nombre d'enfants, statut fumeur et région, "
            "avec leurs frais médicaux annuels."
        )
with col3:
    with st.container(border=True):
        st.markdown("#### :material/model_training: La démarche")
        st.write(
            "Analyse des facteurs de coût, comparaison de quatre modèles de régression, optimisation "
            "du Random Forest, puis simulateur de prime en direct."
        )

st.subheader("L'équipe")
MEMBRES = [
    ("Loïc", "Data Analyst", "Tabagisme"),
    ("Suz", "Data Scientist", "BMI et âge"),
    ("David", "Data Analyst", "Région et famille"),
]
for colonne, (nom, role, lot) in zip(st.columns(3), MEMBRES):
    with colonne:
        with st.container(border=True):
            st.markdown(f"### :material/person: {nom}")
            st.caption(role.upper())
            st.write(lot)
