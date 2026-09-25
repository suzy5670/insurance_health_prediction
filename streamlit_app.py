import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Aegis Health Coverage", page_icon=":material/health_and_safety:", layout="wide")

# Indique au navigateur que la page est en français : évite que la traduction automatique de Chrome ne la casse.
components.html(
    """<script>
    const racine = window.parent.document.documentElement;
    racine.setAttribute("lang", "fr");
    racine.setAttribute("translate", "no");
    </script>""",
    height=0,
)

page = st.navigation({
    "": [
        st.Page("app_pages/accueil.py", title="Accueil", icon=":material/home:"),
    ],
    "Analyse": [
        st.Page("app_pages/analyse_tabagisme.py", title="Tabagisme", icon=":material/smoking_rooms:"),
        st.Page("app_pages/analyse_bmi.py", title="BMI et âge", icon=":material/monitor_weight:"),
        st.Page("app_pages/analyse_region.py", title="Région et famille", icon=":material/map:"),
    ],
    "Modèle et décision": [
        st.Page("app_pages/choix_du_modele.py", title="Choix du modèle", icon=":material/model_training:"),
        st.Page("app_pages/simulateur.py", title="Simulateur", icon=":material/calculate:"),
        st.Page("app_pages/recommandation.py", title="Recommandation", icon=":material/lightbulb:"),
    ],
})
page.run()
