import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, conint, confloat
from enum import Enum
import pandas as pd

rfc = joblib.load("./model/titanic_classifier.joblib")

def predict_titanic_survived(features_passenger: pd.DataFrame, confidence=0.5):
    """Recibe un vector de características de un pasajero a bordo y predice
    si la probabiliad de supervivencia.

    Argumentos:
        features_passenger (pd.DataFrame): Características del pasajero, dataframe de 7 columnas.
        confidence (float, opcional): Nivel de confianza. Por defecto es 0.5.
    """

    pred_value = rfc.predict_proba(features_passenger)[0][1]
    if pred_value >= confidence:
      return "survived"
    else:
      return "no survived"

class Gender(str, Enum):
    male = "male"
    female = "female"

class Embarked(str, Enum):
    emb_C = "C"
    emb_S = "S"
    emb_Q = "Q"


# Asignamos una instancia de la clase FastAPI a la variable "app".
# Interacturaremos con la API usando este elemento.
app = FastAPI(title='Implementando un modelo de Machine Learning usando FastAPI')

# Creamos una clase para el vector de features de entrada
class Item(BaseModel):
    pclass: conint(ge=1, le=3)
    sex: Gender
    age: confloat(gt=0) # edad positiva
    sibsp: conint(ge=0) # 0 hermanos/familiares o más
    parch: conint(ge=0) # 0 hijos o más
    fare: confloat(gt=0) # tarifa positiva
    embarked: Embarked

# Usando @app.get("/") definimos un método GET para el endpoint / (que sería como el "home").
@app.get("/")
def home():
    return "¡Felicitaciones! Tu API está funcionando según lo esperado. Anda ahora a http://localhost:8000/docs."


# Este endpoint maneja la lógica necesaria para clasificar.
# Requiere como entrada el vector de características del viaje y el umbral de confianza para la clasificación.
@app.post("/predict") 
def prediction(item: Item, confidence: float):

    
    # 1. Correr el modelo de clasificación
    features_df = pd.DataFrame({"pclass": [item.pclass], 
                               "sex": [item.sex], 
                               "age": [item.age], 
                               "sibsp": [item.sibsp], 
                               "parch": [item.parch], 
                               "fare": [item.fare], 
                               "embarked": [item.embarked]})

    pred = predict_titanic_survived(features_df, confidence)
    
    # 2. Transmitir la respuesta de vuelta al cliente

    # Retornar el resultado de la predicción
    return {'predicted_class': pred}