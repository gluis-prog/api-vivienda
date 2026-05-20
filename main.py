from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI()

# Permite que cualquier página HTML pueda llamar la API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

modelo = joblib.load("modelo_regresion.pkl")
scaler = joblib.load("feature_scaler.pkl")

FEATURE_COLUMNS = [
    'longitude', 'latitude', 'housing_median_age',
    'total_rooms', 'total_bedrooms', 'population',
    'households', 'median_income',
    'rooms_per_household', 'bedrooms_per_room', 'population_per_household'
]

class DatosVivienda(BaseModel):
    housing_median_age: float
    total_rooms: int
    total_bedrooms: int
    population: int
    households: int
    median_income: float

@app.get("/")
def inicio():
    return {"mensaje": "API Predictor de Vivienda activa"}

@app.post("/predecir")
def predecir(datos: DatosVivienda):
    hh    = datos.households
    rooms = datos.total_rooms
    beds  = datos.total_bedrooms
    pop   = datos.population

    rph = rooms / hh
    bpr = beds  / rooms
    pph = pop   / hh

    # Longitud y latitud fijadas en su media (valor neutral)
    entrada = pd.DataFrame([[
        -119.556526, 35.6177206,
        datos.housing_median_age,
        rooms, beds, pop, hh,
        datos.median_income,
        rph, bpr, pph
    ]], columns=FEATURE_COLUMNS)

    entrada_scaled = scaler.transform(entrada)
    prediccion = float(modelo.predict(entrada_scaled)[0])
    prediccion = max(0, prediccion)

    return {
        "valor_estimado": round(prediccion, 2),
        "valor_formateado": f"${prediccion:,.0f}",
        "variables_derivadas": {
            "habitaciones_por_hogar": round(rph, 2),
            "dormitorios_por_habitacion": round(bpr, 3),
            "personas_por_hogar": round(pph, 2)
        }
    }