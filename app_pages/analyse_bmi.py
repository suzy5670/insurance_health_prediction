import altair as alt
import pandas as pd
import streamlit as st

from utils import BINS_BMI, LABELS_BMI, STATUT, barres_par_statut, charger_donnees, echelle_statut, usd

df = charger_donnees()

st.title("Analyse · BMI et âge")

st.subheader("Le BMI et l'âge amplifient-ils l'effet du tabagisme sur les frais ?")
st.caption("Suz")

bmi_moyen = df["bmi"].mean()
part_obeses = (df["bmi"] >= 30).mean()
with st.container(horizontal=True):
    st.metric("BMI moyen du portefeuille", f"{bmi_moyen:.1f}", border=True)
    st.metric("Assurés en obésité (BMI ≥ 30)", f"{part_obeses:.0%}", border=True)
st.caption(
    "Le BMI (IMC) se calcule ainsi : poids (kg) ÷ taille² (m²). Seuils : 25 = surpoids, 30 = obésité. "
    "Le BMI moyen dépasse déjà le seuil d'obésité."
)

q4, q5, q6 = st.tabs(["BMI et frais", "Vieillissement", "Seuil d'obésité"])

with q4:
    st.markdown("#### 4. Le BMI élevé augmente-t-il les frais de la même façon chez les fumeurs et les non-fumeurs ?")
    nuage = (
        alt.Chart(df)
        .mark_circle(size=45, opacity=0.65)
        .encode(
            x=alt.X("bmi:Q", title="BMI", scale=alt.Scale(zero=False)),
            y=alt.Y("expenses:Q", title="Frais annuels (USD)"),
            color=alt.Color("statut:N", scale=echelle_statut(), legend=alt.Legend(title="Statut")),
            tooltip=[
                alt.Tooltip("statut:N", title="Statut"), alt.Tooltip("age:Q", title="Âge"),
                alt.Tooltip("bmi:Q", title="BMI"), alt.Tooltip("expenses:Q", title="Frais", format=",.0f"),
            ],
        )
    )
    seuil = alt.Chart(pd.DataFrame({"bmi": [30]})).mark_rule(strokeDash=[5, 5], color="#0b0b0b").encode(x="bmi:Q")
    st.altair_chart((nuage + seuil).properties(height=380).interactive())
    corr_bmi = {cle: df.loc[df["smoker"] == cle, "bmi"].corr(df.loc[df["smoker"] == cle, "expenses"]) for cle in STATUT}
    st.info(
        f"La corrélation entre BMI et frais est modérée sur l'ensemble ({df['bmi'].corr(df['expenses']):.2f}), mais elle cache "
        f"deux réalités : **{corr_bmi['yes']:.2f} chez les fumeurs** contre {corr_bmi['no']:.2f} chez les non-fumeurs. "
        "Chez les fumeurs les frais grimpent nettement avec le BMI ; chez les non-fumeurs la relation reste quasi plate. "
        "La ligne pointillée marque le seuil d'obésité (30)."
    )

with q5:
    st.markdown("#### 5. Le vieillissement aggrave-t-il davantage les coûts chez les fumeurs ?")
    tranche_age = pd.cut(
        df["age"], bins=[17, 25, 35, 45, 55, 65], labels=["18-25", "26-35", "36-45", "46-55", "56-65"]
    ).rename("tranche")
    par_age = df.groupby([tranche_age, "statut"], observed=True)["expenses"].mean().reset_index(name="valeur")
    base_age = alt.Chart(par_age).encode(
        x=alt.X("tranche:N", title="Tranche d'âge", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("valeur:Q", title="Frais moyens (USD)"),
        color=alt.Color("statut:N", scale=echelle_statut(), legend=alt.Legend(title="Statut")),
        tooltip=[
            alt.Tooltip("tranche:N", title="Âge"), alt.Tooltip("statut:N", title="Statut"),
            alt.Tooltip("valeur:Q", title="Frais moyens", format=",.0f"),
        ],
    )
    courbes = base_age.mark_line(point=alt.OverlayMarkDef(size=90), strokeWidth=3)
    etiquettes_age = base_age.mark_text(dy=-14, fontWeight="bold").encode(text=alt.Text("valeur:Q", format=",.0f"))
    st.altair_chart((courbes + etiquettes_age).properties(height=380))
    tableau_age = par_age.pivot(index="tranche", columns="statut", values="valeur")
    ecarts_age = tableau_age["Fumeur"] - tableau_age["Non-fumeur"]
    st.info(
        f"Les frais augmentent avec l'âge (corrélation {df['age'].corr(df['expenses']):.2f}) dans les deux groupes, mais "
        f"l'écart entre fumeurs et non-fumeurs reste stable, entre {usd(ecarts_age.min())} et {usd(ecarts_age.max())} selon la "
        "tranche d'âge : le tabagisme est une pénalité quasi constante, pas un effet qui s'accumule avec les années."
    )

with q6:
    st.markdown("#### 6. Existe-t-il un effet de seuil au BMI = 30 qui amplifie le surcoût du tabagisme ?")
    categorie_bmi = pd.cut(df["bmi"], bins=BINS_BMI, labels=LABELS_BMI).rename("categorie")
    par_bmi = df.groupby([categorie_bmi, "statut"], observed=True)["expenses"].mean().reset_index(name="valeur")
    tableau_bmi = par_bmi.pivot(index="categorie", columns="statut", values="valeur").loc[LABELS_BMI]
    saut = tableau_bmi.loc[LABELS_BMI[2], "Fumeur"] / tableau_bmi.loc[LABELS_BMI[1], "Fumeur"] - 1
    hausse = tableau_bmi.loc[LABELS_BMI[2], "Non-fumeur"] / tableau_bmi.loc[LABELS_BMI[0], "Non-fumeur"] - 1
    with st.container(horizontal=True):
        st.metric("Fumeurs : de surpoids à obésité", usd(tableau_bmi.loc[LABELS_BMI[2], "Fumeur"]), f"{saut:+.0%}", border=True)
        st.metric("Non-fumeurs : de normal à obésité", usd(tableau_bmi.loc[LABELS_BMI[2], "Non-fumeur"]), f"{hausse:+.0%}", border=True)
    st.altair_chart(barres_par_statut(par_bmi, "categorie", "valeur", ordre=LABELS_BMI, titre_x="Catégorie de BMI"))
    st.info(
        f"**Insight clé :** l'obésité ne fait quasiment pas varier les frais des non-fumeurs ({hausse:+.0%}), mais elle fait "
        f"exploser ceux des fumeurs (**{saut:+.0%}**). Le tabagisme et l'obésité ne s'additionnent pas : ils se **multiplient**."
    )

st.success(
    "**Conclusion · BMI et âge.** Le BMI amplifie fortement l'effet du tabagisme, l'âge non. "
    "**Aegis devrait tarifer le profil « fumeur + BMI élevé » comme une catégorie de risque à part.**"
)
