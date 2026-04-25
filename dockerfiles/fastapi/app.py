"""
RainTomorrow Prediction API

REST API that serves the Rain Prediction XGBoost classification model.
Loads the model from MLflow model registry and provides prediction endpoints.

Authors: Paola Blanco, Agustín Vazquez, Facundo Quiroga, Victor Peralta
"""

import json
import pickle
import boto3
import mlflow

import numpy as np
import pandas as pd

from typing import Literal
from fastapi import FastAPI, Body, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from typing_extensions import Annotated


def load_model(model_name: str, alias: str):
    """
    Carga el modelo entrenado y el diccionario de datos del pipeline ETL.

    Intenta cargar el modelo desde el MLflow Registry. Si no lo encuentra,
    intenta cargar un modelo fallback local. Si tampoco existe el fallback,
    retorna None sin crashear para que la API pueda arrancar igual.

    También carga los metadatos del pipeline ETL (columnas, parámetros del
    scaler) desde S3/MinIO o desde un archivo local como fallback.

    :param model_name: Nombre del modelo en el MLflow Registry.
    :param alias: Alias de la versión del modelo (ej: "champion").
    :return: Tupla de (modelo, versión, diccionario_de_datos).
             El modelo puede ser None si no está disponible.
    """
    try:
        # Intentar cargar el modelo champion desde MLflow
        mlflow.set_tracking_uri('http://mlflow:5000')
        client_mlflow = mlflow.MlflowClient()

        model_data_mlflow = client_mlflow.get_model_version_by_alias(model_name, alias)
        model_ml = mlflow.sklearn.load_model(model_data_mlflow.source)
        version_model_ml = int(model_data_mlflow.version)
        print(f"[INFO] Modelo cargado desde MLflow. Versión: {version_model_ml}")

    except Exception as e:
        print(f"[WARNING] No se pudo cargar el modelo desde MLflow: {e}")

        # Intentar cargar el modelo fallback local
        try:
            file_ml = open('/app/files/model.pkl', 'rb')
            model_ml = pickle.load(file_ml)
            file_ml.close()
            version_model_ml = 0
            print("[INFO] Modelo cargado desde fallback local.")

        except Exception:
            # Si no hay modelo en MLflow ni fallback local, retornar None.
            # La API arrancará en modo "sin modelo" y avisará al usuario
            # en lugar de crashear el servicio completo.
            print("[WARNING] No se encontró ningún modelo disponible.")
            print("[WARNING] Ejecutar el DAG 'train_initial_model' en Airflow.")
            model_ml = None
            version_model_ml = -1

    try:
        # Cargar metadatos del pipeline ETL desde S3/MinIO
        s3 = boto3.client('s3')
        s3.head_object(Bucket='data', Key='data_info/data.json')
        result_s3 = s3.get_object(Bucket='data', Key='data_info/data.json')
        text_s3 = result_s3["Body"].read().decode()
        data_dictionary = json.loads(text_s3)

        data_dictionary["standard_scaler_mean"] = np.array(
            data_dictionary["standard_scaler_mean"]
        )
        data_dictionary["standard_scaler_std"] = np.array(
            data_dictionary["standard_scaler_std"]
        )
    except Exception:
        # Fallback: cargar metadatos desde archivo local
        try:
            file_s3 = open('/app/files/data.json', 'r')
            data_dictionary = json.load(file_s3)
            file_s3.close()
        except Exception:
            data_dictionary = None

    return model_ml, version_model_ml, data_dictionary


def check_model():
    """
    Verifica si hay una versión más nueva del modelo en MLflow y recarga si es necesario.

    Compara la versión actual con la versión champion en MLflow.
    Si hay una versión nueva, recarga el modelo automáticamente (hot-reload).
    Se ejecuta en background después de cada predicción.
    """
    global model
    global data_dict
    global version_model

    try:
        model_name = "rain_prediction_model_prod"
        alias = "champion"

        mlflow.set_tracking_uri('http://mlflow:5000')
        client = mlflow.MlflowClient()

        new_model_data = client.get_model_version_by_alias(model_name, alias)
        new_version_model = int(new_model_data.version)

        if new_version_model != version_model:
            model, version_model, data_dict = load_model(model_name, alias)
            print(f"[INFO] Modelo actualizado a versión {version_model}")
    except Exception:
        pass


# ============================================
# Pydantic Models for Request/Response validation
# ============================================

class ModelInput(BaseModel):
    """
    Esquema de entrada para el modelo de predicción de lluvia.

    Define las variables meteorológicas requeridas para la predicción,
    correspondientes al dataset weatherAUS.
    """
    Location: str = Field(
        description="Location of the weather station in Australia",
    )
    MinTemp: float = Field(
        description="Minimum temperature in degrees Celsius",
        ge=-20, le=60,
    )
    MaxTemp: float = Field(
        description="Maximum temperature in degrees Celsius",
        ge=-20, le=60,
    )
    Rainfall: float = Field(
        description="Amount of rainfall recorded in mm",
        ge=0, le=400,
    )
    Humidity3pm: float = Field(
        description="Humidity at 3pm (%)",
        ge=0, le=100,
    )
    Pressure3pm: float = Field(
        description="Atmospheric pressure at 3pm (hpa)",
        ge=900, le=1100,
    )
    WindSpeed3pm: float = Field(
        description="Wind speed at 3pm (km/h)",
        ge=0, le=150,
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "Location": "Sydney",
                    "MinTemp": 13.4,
                    "MaxTemp": 22.9,
                    "Rainfall": 0.6,
                    "Humidity3pm": 71,
                    "Pressure3pm": 1015.3,
                    "WindSpeed3pm": 24,
                }
            ]
        }
    }


class ModelOutput(BaseModel):
    """
    Esquema de salida del modelo de predicción de lluvia.

    Devuelve tanto un valor numérico como una descripción legible de la predicción.
    """
    int_output: bool = Field(
        description="Output of the model. True if rain is expected tomorrow",
    )
    str_output: Literal["No rain expected tomorrow", "Rain expected tomorrow"] = Field(
        description="Output of the model in string form",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "int_output": True,
                    "str_output": "Rain expected tomorrow",
                }
            ]
        }
    }


# ============================================
# Carga del modelo al iniciar la API
# ============================================
# Se intenta cargar el modelo champion desde MLflow.
# Si no hay modelo disponible (MLflow vacío y sin fallback local),
# la API arranca igual en modo "sin modelo".
# El endpoint /predict/ avisará al usuario con un mensaje claro
# en lugar de devolver un error 500 inesperado.
model = None
version_model = 0
data_dict = None

try:
    model, version_model, data_dict = load_model("rain_prediction_model_prod", "champion")
except Exception as e:
    print(f"[WARNING] La API arrancó sin modelo disponible: {e}")
    print("[WARNING] Ejecutar el DAG 'train_initial_model' en Airflow.")
    print("[WARNING] Luego llamar a POST /reload/ para cargar el modelo sin reiniciar.")


app = FastAPI(
    title="RainTomorrow Prediction API",
    description="REST API for predicting rain tomorrow using an XGBoost model. "
                "Part of the MLOps1 final project.",
    version="1.0.0",
)


@app.get("/")
async def read_root():
    """
    Endpoint raíz. Confirma que la API está corriendo e informa el estado del modelo.
    """
    estado_modelo = "disponible" if model is not None else "no disponible - ejecutar DAG train_initial_model"
    return JSONResponse(
        content=jsonable_encoder({
            "message": "Welcome to the RainTomorrow Prediction API",
            "model_status": estado_modelo,
            "model_version": version_model if model is not None else None,
        })
    )


@app.post("/predict/", response_model=ModelOutput)
def predict(
    features: Annotated[
        ModelInput,
        Body(embed=True),
    ],
    background_tasks: BackgroundTasks,
):
    """
    Predice si lloverá mañana a partir de variables meteorológicas.

    Si el modelo aún no fue entrenado o registrado en MLflow,
    devuelve un error 503 con instrucciones claras en lugar de crashear.
    """
    # Verificar si el modelo está disponible antes de intentar predecir.
    # Esto ocurre cuando la API arrancó antes de que se corriera el DAG
    # de entrenamiento inicial y no hay modelo champion en MLflow.
    if model is None or data_dict is None:
        return JSONResponse(
            status_code=503,
            content={
                "error": "Modelo no disponible",
                "detail": (
                    "El modelo aún no fue entrenado o registrado en MLflow. "
                    "Por favor seguir estos pasos: "
                    "1) Ejecutar el DAG 'process_etl_rain_data' en Airflow (http://localhost:8080). "
                    "2) Ejecutar el DAG 'train_initial_model' en Airflow. "
                    "3) Llamar a POST /reload/ para cargar el modelo sin reiniciar la API."
                )
            }
        )

    # Convertir input a DataFrame
    features_dict = features.dict()
    features_df = pd.DataFrame([features_dict])

    # Procesar variables categóricas
    for categorical_col in data_dict["categorical_columns"]:
        categories = data_dict["categories_values_per_categorical"][categorical_col]
        features_df[categorical_col] = pd.Categorical(
            features_df[categorical_col], categories=categories
        )

    # Convertir categóricas a variables dummy
    features_df = pd.get_dummies(
        data=features_df,
        columns=data_dict["categorical_columns"],
        drop_first=True,
    )

    # Reordenar columnas para que coincidan con el orden del entrenamiento
    features_df = features_df[data_dict["columns_after_dummy"]]

    # Escalar usando los parámetros del StandardScaler del pipeline ETL
    features_df = (
        features_df - data_dict["standard_scaler_mean"]
    ) / data_dict["standard_scaler_std"]

    # Realizar predicción
    prediction = model.predict(features_df)

    str_pred = "No rain expected tomorrow"
    if prediction[0] > 0:
        str_pred = "Rain expected tomorrow"

    # Verificar actualizaciones del modelo en background (hot-reload)
    background_tasks.add_task(check_model)

    return ModelOutput(int_output=bool(prediction[0].item()), str_output=str_pred)


@app.post("/reload/")
async def reload_model():
    """
    Recarga el modelo champion desde MLflow sin reiniciar la API.

    Útil para cargar el modelo después de correr el DAG de entrenamiento
    sin necesidad de hacer docker restart al contenedor de FastAPI.
    """
    global model, version_model, data_dict
    try:
        model, version_model, data_dict = load_model("rain_prediction_model_prod", "champion")

        if model is None:
            return JSONResponse(
                status_code=503,
                content={
                    "error": "No se encontró ningún modelo",
                    "detail": "Asegurarse de haber corrido los DAGs 'process_etl_rain_data' y 'train_initial_model' en Airflow."
                }
            )

        return JSONResponse(content={
            "message": f"Modelo recargado correctamente.",
            "model_version": version_model,
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": "No se pudo recargar el modelo",
                "detail": str(e)
            }
        )