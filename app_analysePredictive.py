from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import pandas as pd
import numpy as np

# Charger le modèle TensorFlow
MODEL_PATH = "model/analyse_prédictive_30_11.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Charger les données nécessaires
jobs_data = pd.read_csv("data/jobs_data.csv")
job_tokens = np.load("data/job_tokens.npy")

# Définir l'application FastAPI
app = FastAPI()

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

# Définir un schéma pour les données d'entrée
class PredictionRequest(BaseModel):
    features: list[float]  # Liste des caractéristiques d'entrée pour une prédiction unique

@app.post("/predict")
def predict_top_jobs(request: PredictionRequest):
    try:
        # Vérifier la validité des données d'entrée
        if len(request.features) != model.input_shape[1]:
            return {
                "error": f"Le modèle attend {model.input_shape[1]} caractéristiques, mais {len(request.features)} ont été fournies."
            }
        
        # Convertir les caractéristiques en numpy array
        features = np.array(request.features).reshape(1, -1)  # 1 seul exemple

        # Prédire les scores d'intérêt pour tous les jobs
        val_predictions = model.predict(job_tokens).flatten()
        jobs_data['interest_score'] = val_predictions

        # Trier les postes par score d'intérêt et supprimer les doublons
        unique_jobs = jobs_data.sort_values(by='interest_score', ascending=False).drop_duplicates(subset='poste')

        # Obtenir les 5 premiers postes les plus intéressants
        top_interested_jobs = unique_jobs.head(5)

        # Préparer les résultats en JSON
        results = top_interested_jobs[['poste', 'interest_score']].to_dict(orient="records")

        return {
            "top_jobs": results,
            "message": "Prédictions des top postes réalisées avec succès."
        }
    except Exception as e:
        return {"error": str(e)}