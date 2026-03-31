"""
Heart Disease Detector API

REST API that serves the Heart Disease SVM classification model.
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
    Load a trained model and associated data dictionary.

    Attempts to load the model from MLflow registry. If not found,
    falls back to a local default model file. Also loads the ETL
    pipeline metadata (columns, scaler params) from S3 or local file.

    :param model_name: The name of the model in MLflow registry.
    :param alias: The alias of the model version (e.g., "champion").
    :return: Tuple of (model, version, data_dictionary).
    """
    try:
        mlflow.set_tracking_uri('http://mlflow:5000')
        client_mlflow = mlflow.MlflowClient()

        model_data_mlflow = client_mlflow.get_model_version_by_alias(model_name, alias)
        model_ml = mlflow.sklearn.load_model(model_data_mlflow.source)
        version_model_ml = int(model_data_mlflow.version)
    except Exception:
        # If there is no registry in MLflow, open the default model
        file_ml = open('/app/files/model.pkl', 'rb')
        model_ml = pickle.load(file_ml)
        file_ml.close()
        version_model_ml = 0

    try:
        # Load information of the ETL pipeline from S3
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
        # If data dictionary is not found in S3, load from local file
        file_s3 = open('/app/files/data.json', 'r')
        data_dictionary = json.load(file_s3)
        file_s3.close()

    return model_ml, version_model_ml, data_dictionary


def check_model():
    """
    Check for updates in the model registry and hot-reload if needed.

    Compares the current model version with the champion version in MLflow.
    If versions differ, reloads the model, version, and data dictionary.
    """
    global model
    global data_dict
    global version_model

    try:
        model_name = "heart_disease_model_prod"
        alias = "champion"

        mlflow.set_tracking_uri('http://mlflow:5000')
        client = mlflow.MlflowClient()

        new_model_data = client.get_model_version_by_alias(model_name, alias)
        new_version_model = int(new_model_data.version)

        if new_version_model != version_model:
            model, version_model, data_dict = load_model(model_name, alias)
    except Exception:
        pass


# ============================================
# Pydantic Models for Request/Response validation
# ============================================

class ModelInput(BaseModel):
    """
    Input schema for the heart disease prediction model.

    Defines the 13 clinical features required for prediction,
    with validation constraints matching the Heart Disease dataset.
    """
    age: int = Field(
        description="Age of the patient",
        ge=0, le=150,
    )
    sex: int = Field(
        description="Sex of the patient. 1: male; 0: female",
        ge=0, le=1,
    )
    cp: int = Field(
        description="Chest pain type. 1: typical angina; 2: atypical angina; "
                    "3: non-anginal pain; 4: asymptomatic",
        ge=1, le=4,
    )
    trestbps: float = Field(
        description="Resting blood pressure in mm Hg on admission to the hospital",
        ge=90, le=220,
    )
    chol: float = Field(
        description="Serum cholesterol in mg/dl",
        ge=110, le=600,
    )
    fbs: int = Field(
        description="Fasting blood sugar. 1: >120 mg/dl; 0: <120 mg/dl",
        ge=0, le=1,
    )
    restecg: int = Field(
        description="Resting electrocardiographic results. "
                    "0: normal; 1: ST-T wave abnormality; 2: left ventricular hypertrophy",
        ge=0, le=2,
    )
    thalach: float = Field(
        description="Maximum heart rate achieved (beats per minute)",
        ge=50, le=210,
    )
    exang: int = Field(
        description="Exercise induced angina. 1: yes; 0: no",
        ge=0, le=1,
    )
    oldpeak: float = Field(
        description="ST depression induced by exercise relative to rest",
        ge=0.0, le=7.0,
    )
    slope: int = Field(
        description="The slope of the peak exercise ST segment. "
                    "1: upsloping; 2: flat; 3: downsloping",
        ge=1, le=3,
    )
    ca: int = Field(
        description="Number of major vessels colored by fluoroscopy",
        ge=0, le=3,
    )
    thal: Literal[3, 6, 7] = Field(
        description="Thalassemia. 3: normal; 6: fixed defect; 7: reversible defect",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 67,
                    "sex": 1,
                    "cp": 4,
                    "trestbps": 160.0,
                    "chol": 286.0,
                    "fbs": 0,
                    "restecg": 2,
                    "thalach": 108.0,
                    "exang": 1,
                    "oldpeak": 1.5,
                    "slope": 2,
                    "ca": 3,
                    "thal": 3,
                }
            ]
        }
    }


class ModelOutput(BaseModel):
    """
    Output schema for the heart disease prediction model.

    Returns both a boolean and human-readable string for the prediction.
    """
    int_output: bool = Field(
        description="Output of the model. True if the patient has heart disease",
    )
    str_output: Literal["Healthy patient", "Heart disease detected"] = Field(
        description="Output of the model in string form",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "int_output": True,
                    "str_output": "Heart disease detected",
                }
            ]
        }
    }


# ============================================
# Load model at startup
# ============================================
model, version_model, data_dict = load_model("heart_disease_model_prod", "champion")

app = FastAPI(
    title="Heart Disease Detector API",
    description="REST API for predicting heart disease using an SVM model. "
                "Part of the MLOps1 final project.",
    version="1.0.0",
)


@app.get("/")
async def read_root():
    """
    Root endpoint. Returns a welcome message confirming the API is running.
    """
    return JSONResponse(
        content=jsonable_encoder(
            {"message": "Welcome to the Heart Disease Detector API"}
        )
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
    Predict whether a patient has heart disease.

    Receives 13 clinical features and returns the prediction
    as both a boolean and a descriptive string.
    """
    # Extract features and convert to DataFrame
    features_list = [*features.dict().values()]
    features_key = [*features.dict().keys()]

    features_df = pd.DataFrame(
        np.array(features_list).reshape([1, -1]),
        columns=features_key
    )

    # Process categorical features
    for categorical_col in data_dict["categorical_columns"]:
        features_df[categorical_col] = features_df[categorical_col].astype(int)
        categories = data_dict["categories_values_per_categorical"][categorical_col]
        features_df[categorical_col] = pd.Categorical(
            features_df[categorical_col], categories=categories
        )

    # Convert categorical features into dummy variables
    features_df = pd.get_dummies(
        data=features_df,
        columns=data_dict["categorical_columns"],
        drop_first=True,
    )

    # Reorder DataFrame columns to match training order
    features_df = features_df[data_dict["columns_after_dummy"]]

    # Scale using the fitted StandardScaler parameters
    features_df = (
        features_df - data_dict["standard_scaler_mean"]
    ) / data_dict["standard_scaler_std"]

    # Make prediction
    prediction = model.predict(features_df)

    # Convert to string output
    str_pred = "Healthy patient"
    if prediction[0] > 0:
        str_pred = "Heart disease detected"

    # Check for model updates in the background
    background_tasks.add_task(check_model)

    return ModelOutput(int_output=bool(prediction[0].item()), str_output=str_pred)
