from fastapi import FastAPI
import os
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, Field
import joblib
import numpy as np
# Charger les variables d'environnement
load_dotenv()

# Client Groq (chargé au démarrage)
groq_client = None

groq_api_key = os.getenv("GROQ_API_KEY")

if groq_api_key:
    groq_client = Groq(api_key=groq_api_key)
    print("Client Groq initialise.")
else:
    print("ATTENTION : GROQ_API_KEY non trouvée. /explain sera désactivé.")
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
class ExplainInput(BaseModel):
    diagnostic: str = Field(
        ...,
        description="Diagnostic prédit par le modèle"
    )

    probabilite: float = Field(
        ...,
        description="Probabilité du diagnostic"
    )

    age: int = Field(...)
    sexe: str = Field(...)
    temperature: float = Field(...)
    region: str = Field(...)


class ExplainOutput(BaseModel):
    explication: str = Field(
        ...,
        description="Explication en français"
    )

    modele_llm: str = Field(
        default="llama-3.1-8b-instant",
        description="Modèle LLM utilisé"
    )

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
SYSTEM_PROMPT = """
Tu es un assistant médical sénégalais.

Tu reçois un diagnostic et des données patient.
Explique le résultat en français simple,
comme un médecin parlerait à son patient.

Sois rassurant mais recommande toujours
une consultation médicale.

Maximum 3 phrases.

Ne fais JAMAIS de diagnostic toi-même.
Tu expliques uniquement le diagnostic fourni.
"""


@app.post("/explain", response_model=ExplainOutput)
def explain(data: ExplainInput):
    """Expliquer un diagnostic en français avec un LLM."""

    if not groq_client:
        return ExplainOutput(
            explication="Service d'explication indisponible. Clé API non configurée.",
            modele_llm="aucun"
        )

    # Construire le prompt
    user_prompt = (
        f"Patient : {data.sexe}, {data.age} ans, "
        f"region {data.region}\n"
        f"Temperature : {data.temperature} C\n"
        f"Diagnostic du modele : {data.diagnostic} "
        f"(probabilite {data.probabilite:.0%})\n"
        f"Explique ce resultat au patient."
    )

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            max_tokens=200,
            temperature=0.3
        )

        explication = response.choices[0].message.content

    except Exception as e:
        explication = f"Erreur lors de l'appel au LLM : {str(e)}"

    return ExplainOutput(explication=explication)