from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np

app = FastAPI(
    title="SenSante API",
    description="Assistant pre-diagnostic medical",
    version="0.2.0"
)
from fastapi . middleware . cors import CORSMiddleware
# Autoriser les requêtes depuis le frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== MODELE =====
print("Chargement du modele...")
model = joblib.load("models/model.pkl")
le_sexe = joblib.load("models/encoder_sexe.pkl")
le_region = joblib.load("models/encoder_region.pkl")

print("Modele charge :", model.classes_)


# ===== SCHEMAS =====
class PatientInput(BaseModel):
    age: int = Field(..., ge=0, le=120)
    sexe: str
    temperature: float = Field(..., ge=35.0, le=42.0)
    tension_sys: int = Field(..., ge=60, le=250)
    toux: bool
    fatigue: bool
    maux_tete: bool
    region: str


class DiagnosticOutput(BaseModel):
    diagnostic: str
    probabilite: float
    confiance: str
    message: str


# ===== ROUTES =====
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=DiagnosticOutput)
def predict(patient: PatientInput):

    sexe_enc = le_sexe.transform([patient.sexe])[0]
    region_enc = le_region.transform([patient.region])[0]

    features = np.array([[
        patient.age,
        sexe_enc,
        patient.temperature,
        patient.tension_sys,
        int(patient.toux),
        int(patient.fatigue),
        int(patient.maux_tete),
        region_enc
    ]])

    pred = model.predict(features)[0]
    proba = float(model.predict_proba(features).max())

    if proba >= 0.7:
        conf = "haute"
    elif proba >= 0.4:
        conf = "moyenne"
    else:
        conf = "faible"

    messages = {
        "palu": "Suspicion paludisme",
        "grippe": "Suspicion grippe",
        "typh": "Suspicion typhoide",
        "sain": "Etat normal"
    }

    return DiagnosticOutput(
        diagnostic=pred,
        probabilite=round(proba, 2),
        confiance=conf,
        message=messages.get(pred, "Consultez un medecin")
    )