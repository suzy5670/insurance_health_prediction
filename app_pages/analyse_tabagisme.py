import altair as alt
import streamlit as st

from utils import barres_par_statut, charger_donnees, echelle_statut, usd

df = charger_donnees()

st.title("Analyse · Tabagisme")

st.subheader("Le tabagisme est-il le facteur qui influence le plus les frais médicaux, et de combien ?")
st.caption("Loïc")

cout_moyen = df["expenses"].mean()
cout_median = df["expenses"].median()
part_fumeurs = (df["smoker"] == "yes").mean()
with st.container(horizontal=True):
    st.metric("Coût moyen du portefeuille", usd(cout_moyen), border=True)
    st.metric("Coût médian du portefeuille", usd(cout_median), border=True)
    st.metric("Part de fumeurs", f"{part_fumeurs:.1%}", border=True)
st.caption("La moyenne dépasse la médiane : quelques assurés très coûteux tirent la moyenne vers le haut.")

q1, q2, q3 = st.tabs(["Distribution des frais", "Écart fumeurs / non-fumeurs", "Effet du sexe"])

with q1:
    st.markdown("#### 1. La distribution des frais révèle-t-elle un sous-groupe à part (les fumeurs) ?")
    histogramme = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("expenses:Q", bin=alt.Bin(maxbins=40), title="Frais annuels (USD)"),
            y=alt.Y("count():Q", title="Nombre d'assurés"),
            color=alt.Color("statut:N", scale=echelle_statut(), legend=alt.Legend(title="Statut")),
            tooltip=[alt.Tooltip("statut:N", title="Statut"), alt.Tooltip("count():Q", title="Assurés")],
        )
        .properties(height=350)
    )
    st.altair_chart(histogramme)
    frais_eleves = df[df["expenses"] > 30_000]
    st.info(
        f"**Réponse : oui.** Les non-fumeurs sont concentrés sous ≈ 15 000 USD, tandis que les fumeurs forment un groupe "
        f"à part, étalé de ≈ 15 000 à 60 000 USD. Sur les {len(frais_eleves)} assurés à plus de 30 000 USD, "
        f"**{(frais_eleves['smoker'] == 'yes').mean():.0%} sont fumeurs**."
    )

with q2:
    st.markdown("#### 2. Quel est l'écart de coût chiffré entre fumeurs et non-fumeurs ?")
    stats = df.groupby("statut")["expenses"].agg(moyenne="mean", mediane="median")
    ecart_moyen = stats.loc["Fumeur", "moyenne"] - stats.loc["Non-fumeur", "moyenne"]
    ratio = stats.loc["Fumeur", "moyenne"] / stats.loc["Non-fumeur", "moyenne"]
    part_depenses = df.loc[df["smoker"] == "yes", "expenses"].sum() / df["expenses"].sum()
    with st.container(horizontal=True):
        st.metric("Écart de coût moyen", "+" + usd(ecart_moyen), border=True)
        st.metric("Un fumeur coûte", f"{ratio:.1f} fois plus", border=True)
        st.metric("Part des dépenses des fumeurs", f"{part_depenses:.0%}", border=True)
    moyenne_mediane = (
        stats.rename(columns={"moyenne": "Moyenne", "mediane": "Médiane"})
        .reset_index()
        .melt(id_vars="statut", var_name="mesure", value_name="valeur")
    )
    st.altair_chart(barres_par_statut(moyenne_mediane, "mesure", "valeur", ordre=["Moyenne", "Médiane"]))
    st.info(
        f"Un fumeur coûte en moyenne {usd(stats.loc['Fumeur', 'moyenne'])} contre {usd(stats.loc['Non-fumeur', 'moyenne'])} "
        f"pour un non-fumeur. Les fumeurs ne représentent que {part_fumeurs:.0%} des assurés mais {part_depenses:.0%} "
        "des dépenses totales : cet écart justifie une tarification différenciée."
    )

with q3:
    st.markdown("#### 3. Ce surcoût est-il le même pour les hommes et les femmes ?")
    par_sexe = (
        df.assign(sexe=df["sex"].map({"female": "Femmes", "male": "Hommes"}))
        .groupby(["sexe", "statut"])["expenses"]
        .mean()
        .reset_index(name="valeur")
    )
    st.altair_chart(barres_par_statut(par_sexe, "sexe", "valeur"))
    tableau_sexe = par_sexe.pivot(index="sexe", columns="statut", values="valeur")
    surcout = tableau_sexe["Fumeur"] - tableau_sexe["Non-fumeur"]
    st.info(
        f"Le surcoût du tabagisme existe pour les deux sexes et reste du même ordre de grandeur : +{usd(surcout['Femmes'])} "
        f"chez les femmes et +{usd(surcout['Hommes'])} chez les hommes. Le sexe seul ne change presque rien : chez les "
        f"non-fumeurs, femmes et hommes coûtent à peu près pareil ({usd(tableau_sexe.loc['Femmes', 'Non-fumeur'])} "
        f"contre {usd(tableau_sexe.loc['Hommes', 'Non-fumeur'])})."
    )

st.success(
    f"**Conclusion · Tabagisme.** Le tabagisme est de loin le facteur qui influence le plus les frais : un fumeur ajoute "
    f"≈ {usd(ecart_moyen)} par an (×{ratio:.1f}), pour les hommes comme pour les femmes. "
    "**Pour Aegis, le statut fumeur doit être le premier critère de tarification.**"
)
