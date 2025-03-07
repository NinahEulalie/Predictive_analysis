import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf

# Charger le modèle TensorFlow
MODEL_PATH = "model/analyse_prédictive_30_11.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Charger les données nécessaires
jobs_data = pd.read_csv("data/jobs_data.csv")
job_tokens = np.load("data/job_tokens.npy")

# Fonction pour prédire les top jobs
def predict_top_jobs():
    try:
        # Faire les prédictions pour tous les jobs
        val_predictions = model.predict(job_tokens).flatten()
        jobs_data['interest_score'] = val_predictions

        # Trier les jobs par score d'intérêt et supprimer les doublons
        unique_jobs = jobs_data.sort_values(by='interest_score', ascending=False).drop_duplicates(subset='poste')

        # Obtenir les 5 meilleurs jobs
        top_interested_jobs = unique_jobs.head(5)

        # Préparer les résultats
        results = top_interested_jobs[['poste', 'interest_score']]
        return results
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {str(e)}")
        return None

# Interface utilisateur Streamlit
st.title("Analyse Prédictive des Intérêts des Visiteurs")
st.subheader("Top 5 des postes les plus susceptibles d'intéresser les visiteurs")

# Bouton pour lancer les prédictions
if st.button("Afficher les résultats"):
    top_jobs = predict_top_jobs()
    
    if top_jobs is not None:
        st.success("Prédictions réalisées avec succès !")
        st.write(top_jobs)
    else:
        st.success("Aucune prédiction à afficher !")
