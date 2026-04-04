"""
RainTomorrow Prediction API

REST API that serves the Rain Prediction SVM classification model.
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
        model_name = "rain_prediction_model_prod"
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
    Input schema for the rain prediction model.

    Defines the meteorological features required for prediction,
    matching the weatherAUS dataset.
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
    Output schema for the rain prediction model.

    Returns both a numeric and human-readable string for the prediction.
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
# Load model at startup
# ============================================
model, version_model, data_dict = load_model("rain_prediction_model_prod", "champion")

app = FastAPI(
    title="RainTomorrow Prediction API",
    description="REST API for predicting rain tomorrow using an SVM model. "
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
            {"message": "Welcome to the RainTomorrow Prediction API"}
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
    Predict whether it will rain tomorrow.

    Receives meteorological features and returns the prediction
    as both a boolean and a descriptive string.
    """
    # Extract features and convert to DataFrame
    features_dict = features.dict()
    features_df = pd.DataFrame([features_dict])

    # Process categorical features
    for categorical_col in data_dict["categorical_columns"]:
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
    str_pred = "No rain expected tomorrow"
    if prediction[0] > 0:
        str_pred = "Rain expected tomorrow"

    # Check for model updates in the background
    background_tasks.add_task(check_model)

    return ModelOutput(int_output=bool(prediction[0].item()), str_output=str_pred)
