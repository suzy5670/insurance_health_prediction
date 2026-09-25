import altair as alt
import pandas as pd
import streamlit as st

from utils import COULEUR_NEUTRE, LIBELLES, REGIONS, barres_par_statut, charger_donnees, usd

df = charger_donnees()

st.title("Analyse · Région et famille")

# écart moyen fumeur / non-fumeur, le même calcul que dans la page Tabagisme
moyennes = df.groupby("statut")["expenses"].mean()
ecart_moyen = moyennes["Fumeur"] - moyennes["Non-fumeur"]

st.subheader("Les caractéristiques démographiques justifient-elles une différenciation tarifaire face au tabagisme et au BMI ?")
st.caption("David")

frais_region = df.groupby("region")["expenses"].mean().sort_values(ascending=False)
with st.container(horizontal=True):
    for region, valeur in frais_region.items():
        st.metric(f"Frais moyens · {REGIONS[region]}", usd(valeur), f"{valeur / df['expenses'].mean() - 1:+.0%} vs portefeuille",
                  delta_color="off", border=True)
ecart_regions = frais_region.max() - frais_region.min()
st.caption(
    f"L'écart maximal entre régions ({usd(ecart_regions)}) reste faible face à l'écart fumeur / non-fumeur "
    f"(≈ {usd(ecart_moyen)})."
)

q7, q8, q9 = st.tabs(["Région", "Taille du foyer", "Variables du modèle"])

with q7:
    st.markdown("#### 7. Le lieu de résidence justifie-t-il une différenciation régionale ?")
    par_region = (
        df.assign(region_fr=df["region"].map(REGIONS))
        .groupby(["region_fr", "statut"])["expenses"]
        .mean()
        .reset_index(name="valeur")
    )
    ordre_regions = [REGIONS[r] for r in frais_region.sort_values().index]
    st.altair_chart(barres_par_statut(par_region, "region_fr", "valeur", ordre=ordre_regions, titre_x="Région"))
    profil = df.groupby("region").agg(
        frais_moyens=("expenses", "mean"), part_fumeurs=("smoker", lambda s: (s == "yes").mean()), bmi_moyen=("bmi", "mean")
    ).sort_values("frais_moyens", ascending=False)
    profil.index = profil.index.map(REGIONS)
    st.dataframe(
        profil,
        column_config={
            "frais_moyens": st.column_config.NumberColumn("Frais moyens (USD)", format="%.0f"),
            "part_fumeurs": st.column_config.NumberColumn("Part de fumeurs", format="percent"),
            "bmi_moyen": st.column_config.NumberColumn("BMI moyen", format="%.1f"),
        },
    )
    region_chere = frais_region.idxmax()
    st.info(
        f"La région la plus chère ({REGIONS[region_chere]}) compte {profil.loc[REGIONS[region_chere], 'part_fumeurs']:.0%} de "
        f"fumeurs et un BMI moyen de {profil.loc[REGIONS[region_chere], 'bmi_moyen']:.1f} : l'écart vient surtout du **profil des "
        "assurés**, pas du lieu de résidence. Une tarification régionale n'est pas prioritaire."
    )

with q8:
    st.markdown("#### 8. La taille du foyer a-t-elle un impact significatif ?")
    foyer = df.groupby("children")["expenses"].agg(effectif="count", moyenne="mean").reset_index()
    base_foyer = alt.Chart(foyer).encode(
        x=alt.X("children:O", title="Nombre d'enfants", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("moyenne:Q", title="Frais moyens (USD)"),
        tooltip=[
            alt.Tooltip("children:O", title="Enfants"), alt.Tooltip("effectif:Q", title="Assurés"),
            alt.Tooltip("moyenne:Q", title="Frais moyens", format=",.0f"),
        ],
    )
    barres_foyer = base_foyer.mark_bar(color=COULEUR_NEUTRE)
    etiquettes_foyer = base_foyer.mark_text(dy=-8, fontWeight="bold").encode(text=alt.Text("moyenne:Q", format=",.0f"))
    st.altair_chart((barres_foyer + etiquettes_foyer).properties(height=350))
    effectifs = foyer.set_index("children")["effectif"]
    st.info(
        f"Les groupes à 4 et 5 enfants sont très petits ({int(effectifs[4])} et {int(effectifs[5])} assurés) : "
        f"leurs moyennes ne sont pas fiables. La corrélation entre enfants et frais est de {df['children'].corr(df['expenses']):.2f} : "
        "**effet faible, le nombre d'enfants n'est pas un critère de tarification.**"
    )

with q9:
    st.markdown("#### 9. Quelles variables doivent réellement entrer dans le modèle ?")
    df_num = df.drop(columns=["statut"]).copy()
    df_num["smoker"] = (df_num["smoker"] == "yes").astype(int)
    df_num["sex"] = (df_num["sex"] == "male").astype(int)
    df_num = pd.get_dummies(df_num, columns=["region"], drop_first=True, dtype=int)
    corr = df_num.corr().rename(index=LIBELLES, columns=LIBELLES)
    ordre_corr = list(corr.columns)
    corr_long = corr.reset_index(names="ligne").melt(id_vars="ligne", var_name="colonne", value_name="correlation")
    carte = alt.Chart(corr_long).encode(
        x=alt.X("colonne:N", sort=ordre_corr, title=None, axis=alt.Axis(labelAngle=-30)),
        y=alt.Y("ligne:N", sort=ordre_corr, title=None),
    )
    carres = carte.mark_rect().encode(
        color=alt.Color("correlation:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1], reverse=True), title="Corrélation"),
        tooltip=[alt.Tooltip("ligne:N", title="Variable"), alt.Tooltip("colonne:N", title="Avec"),
                 alt.Tooltip("correlation:Q", title="Corrélation", format=".2f")],
    )
    valeurs_corr = carte.mark_text(fontSize=11).encode(
        text=alt.Text("correlation:Q", format=".2f"),
        color=alt.condition("abs(datum.correlation) > 0.5", alt.value("white"), alt.value("black")),
    )
    st.altair_chart((carres + valeurs_corr).properties(height=450))
    st.info(
        f"Le statut fumeur domine ({corr.loc['Frais', 'Fumeur']:.2f}), suivi de l'âge ({corr.loc['Frais', 'Âge']:.2f}) et du BMI "
        f"({corr.loc['Frais', 'BMI']:.2f}). Sexe, enfants et région pèsent très peu. Le BMI reste indispensable malgré sa "
        "corrélation modérée : son effet passe par l'interaction avec le tabagisme, que la corrélation seule ne voit pas."
    )

st.success(
    "**Conclusion · Région et famille.** Les caractéristiques démographiques ne justifient pas de différenciation tarifaire "
    f"face au tabagisme et au BMI : la région crée un écart d'environ {usd(ecart_regions)}, expliqué surtout par le profil "
    "des assurés. **Aegis doit tarifer d'abord sur le statut fumeur, le BMI et l'âge.**"
)
