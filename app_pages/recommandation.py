import streamlit as st

from utils import charger_donnees, evaluer_modele, indicateurs, usd

df = charger_donnees()
performances = evaluer_modele()
chiffres = indicateurs()
nb_assures = f"{len(df):,}".replace(",", " ")

st.title("Recommandation")
st.caption("Ce qu'Aegis peut décider à partir de l'analyse et du modèle.")

with st.container(border=True):
    st.markdown("### :material/sell: Politique de tarification")
    st.markdown(
        f"1. **Le statut fumeur est le premier critère.** Un fumeur coûte en moyenne {usd(chiffres['ecart'])} de plus par an "
        f"(×{chiffres['ratio']:.1f}) et les fumeurs génèrent {chiffres['part_depenses_fumeurs']:.0%} des dépenses alors qu'ils sont "
        f"{(df['smoker'] == 'yes').mean():.0%} des assurés.\n"
        f"2. **Créer une catégorie de risque « fumeur + BMI ≥ 30 ».** Chez les fumeurs, passer du surpoids à l'obésité augmente "
        f"les frais de {chiffres['saut_fumeurs']:.0%}, contre {chiffres['hausse_non_fumeurs']:.0%} chez les non-fumeurs : les deux "
        "facteurs se multiplient.\n"
        "3. **Ajuster ensuite selon le BMI et l'âge**, de façon progressive : les frais augmentent régulièrement avec l'âge, dans "
        "les deux groupes.\n"
        f"4. **Ne pas différencier selon le sexe, la région ou le nombre d'enfants.** La région crée un écart d'environ "
        f"{usd(chiffres['ecart_regions'])}, expliqué surtout par le profil des assurés, et ces trois variables pèsent très "
        "peu dans le modèle."
    )

with st.container(border=True):
    st.markdown("### :material/warning: Limites et déploiement")
    st.markdown(
        f"1. **Une précision correcte, pas parfaite.** Le modèle explique {performances['R2']:.0%} de la variabilité des frais "
        f"avec une erreur moyenne de {usd(performances['MAE'])} par assuré : il sert à cadrer une prime, pas à la fixer seul.\n"
        "2. **Les coûts extrêmes sont sous-estimés.** Les plus grosses erreurs sont des assurés dont les frais dépassent "
        "largement la prédiction : prévoir une marge de sécurité ou une revue humaine pour les cas atypiques.\n"
        f"3. **Peu de données et peu de variables.** {nb_assures} assurés et six variables seulement, sans antécédents "
        "médicaux ni historique de sinistres : enrichir les données améliorerait le modèle.\n"
        "4. **Un outil d'aide à la décision.** Déployer le simulateur auprès des souscripteurs, suivre l'écart entre frais "
        "prédits et frais réels, et ré-entraîner le modèle régulièrement."
    )

st.success(
    "**En résumé :** tarifer d'abord sur le statut fumeur, isoler le profil « fumeur + BMI élevé », ajuster selon le BMI et "
    "l'âge, et utiliser le modèle comme aide à la décision plutôt que comme prix automatique."
)
