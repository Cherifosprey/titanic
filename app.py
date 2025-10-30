import streamlit as st
import pandas as pd
import joblib

# --- Configuration de la Page ---
st.set_page_config(
    page_title="Titanic Predictor",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- Chargement du modèle et du scaler ---
try:
    # Les fichiers de modèle à 3 features (Sex, Age, Pclass)
    model = joblib.load('model_simple.pkl')
    scaler = joblib.load('scaler_simple.pkl')
except:
    st.error("Erreur critique: Fichiers de modèle introuvables. Assurez-vous d'avoir 'model_simple.pkl' et 'scaler_simple.pkl' dans le même dossier.")
    st.stop()


# --- Mappage pour les entrées ---
sex_map = {'Homme': 0, 'Femme': 1}
sex_map_display = {0: 'Homme', 1: 'Femme'} 

# --- Interface Utilisateur ---
st.markdown("<h1 style='text-align: center;'>Prédiction de Survie Titanic</h1>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("Modèle de Régression Logistique basé sur **Sexe**, **Âge**, et **Classe**.")

st.header("Caractéristiques du Passager")

# Utilisation des colonnes pour aligner les champs d'entrée
col1, col2, col3 = st.columns(3)

with col1:
    sex_label = st.selectbox('Sexe', ['Homme', 'Femme'], index=0)
    sex = sex_map[sex_label]

with col2:
    age = st.slider('Âge', 0, 80, 28)

with col3:
    pclass = st.selectbox('Classe du Billet (Pclass)', [1, 2, 3], index=2)


st.markdown("---")

# Bouton de Prédiction
col_center = st.columns([1, 2, 1])
with col_center[1]:
    predict_button = st.button('Prédire la Survie', use_container_width=True)


# --- Logique de Prédiction ---
if predict_button:
    
    # Préparation des données dans l'ordre: ['Sex', 'Age', 'Pclass']
    features = [sex, age, pclass]
    
    # Mise à l'échelle
    features_scaled = scaler.transform([features])
    
    # Prédiction
    prediction = model.predict(features_scaled)
    prediction_proba = model.predict_proba(features_scaled)

    # Affichage du Résultat
    st.subheader("Résultat")
    
    if prediction[0] == 1:
        st.success('PRÉDICTION : SURVIVANT')
        probabilite = prediction_proba[0][1] * 100
    else:
        st.error('PRÉDICTION : NON-SURVIVANT')
        probabilite = prediction_proba[0][0] * 100
        
    st.metric(label="Probabilité", value=f"{probabilite:.2f} %")
    
    with st.expander("Détails des entrées"):
        data_entered = {
            'Caractéristique': ['Sexe', 'Âge', 'Classe'],
            'Valeur': [sex_map_display[sex], age, pclass]
        }
        st.table(pd.DataFrame(data_entered))