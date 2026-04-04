# Proyecto Final - MLOps1: RainTomorrow Prediction Model

### Operaciones de Aprendizaje Automático I - CEIA - FIUBA

**Equipo:**
- Paola Andrea Blanco (a2303)
- Agustín Jesús Vazquez (e2301)
- Facundo Manuel Quiroga (a2305)
- Victor Gabriel Peralta (a2322)

**Profesor:** Facundo Lucianna

---

## Descripción

Implementación productiva de un modelo de machine learning para la predicción de lluvia al día siguiente (`RainTomorrow`), utilizando la infraestructura de **ML Models and something more Inc.**

El modelo fue originalmente desarrollado en la materia **Aprendizaje de Máquina I** sobre el dataset *Rain in Australia*, y en este proyecto se lo lleva a un entorno productivo completo incorporando prácticas de **MLOps**:

- Orquestación de pipelines con Apache Airflow  
- Tracking de experimentos y modelos con MLflow  
- Data Lake basado en S3 (MinIO)  
- Servicio de predicción mediante API REST con FastAPI  

## Problema de Negocio

El objetivo es predecir si lloverá al día siguiente en distintas localidades de Australia, a partir de variables meteorológicas actuales.

Este tipo de modelo puede utilizarse para:
- planificación agrícola  
- logística y transporte  
- toma de decisiones en energía y recursos  

---

## Dataset

- **Fuente:** Rain in Australia (Kaggle)
- **Observaciones:** ~145.000
- **Variables:** 23
- **Variable objetivo:** `RainTomorrow` (Yes / No)
- **Desbalance:** ~77% No / 23% Yes

---

## Arquitectura de Servicios

| Servicio | Puerto | Descripción |
|---|---|---|
| Apache Airflow | [localhost:8080](http://localhost:8080) | Orquestación de pipelines ETL y reentrenamiento |
| MLflow | [localhost:5001](http://localhost:5001) | Tracking de experimentos y registro de modelos |
| MinIO (S3) | [localhost:9001](http://localhost:9001) | Data Lake (almacenamiento de datos y artefactos) |
| FastAPI | [localhost:8800](http://localhost:8800) | REST API para servir predicciones |
| API Docs | [localhost:8800/docs](http://localhost:8800/docs) | Documentación interactiva de la API (Swagger) |
| PostgreSQL | localhost:5432 | Base de datos compartida (Airflow + MLflow) |

---

## Arquitectura del Pipeline

El flujo completo del sistema es el siguiente:

Raw → ETL → Train → MLflow → API

### ETL Pipeline (Airflow)
- Extracción de datos
- Limpieza de valores faltantes
- Encoding de variables categóricas
- Escalado de variables numéricas
- Split train/test

### Training Pipeline
- Entrenamiento de múltiples modelos
- Optimización de hiperparámetros
- Evaluación de métricas
- Registro en MLflow

### Model Serving
- Exposición del modelo mediante FastAPI
- Endpoint `/predict`

---

## Estructura del Proyecto

```
.
├── docker-compose.yaml          # Definición de todos los servicios
├── .env                          # Variables de configuración
├── .gitignore
├── README.md
├── CONTRIBUTING.md
│
├── airflow/
│   ├── dags/
│   │   ├── etl_rain_process.py       # DAG: Pipeline ETL (fetch → clean → split → normalize)
│   │   └── retrain_rain_model.py     # DAG: Reentrenamiento y comparación con champion
│   ├── secrets/
│   │   ├── variables.yaml        # Variables de Airflow
│   │   └── connections.yaml      # Conexiones de Airflow
│   ├── config/
│   ├── logs/
│   └── plugins/
│
├── dockerfiles/
│   ├── airflow/                  # Imagen custom de Airflow
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── mlflow/                   # Imagen custom de MLflow
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── postgres/                 # Imagen custom de PostgreSQL
│   │   ├── Dockerfile
│   │   └── mlflow.sql
│   └── fastapi/                  # Imagen custom de FastAPI
│       ├── Dockerfile
│       ├── requirements.txt
│       ├── app.py                # Aplicación FastAPI
│       └── files/                # Fallback files (modelo y datos por defecto)
│           └── data.json
│
└── notebook_example/
    └── hyperparameter_search.ipynb  # Búsqueda de hiperparámetros con MLflow
```

## Instalación y Uso

### Requisitos Previos
- [Docker](https://docs.docker.com/engine/install/) instalado
- Al menos 4GB de RAM disponible para Docker
- Al menos 2 CPUs

### 1. Levantar los servicios

```bash
docker compose --profile all up
```

Esperar hasta que todos los servicios estén healthy (verificar con `docker ps -a`).

### 2. Ejecutar el pipeline ETL

1. Acceder a Airflow: [http://localhost:8080](http://localhost:8080) (usuario: `airflow`, contraseña: `airflow`)
2. Activar y ejecutar el DAG `process_etl_rain_data`
3. Esto creará los datos procesados en el bucket `s3://data`

### 3. Búsqueda de hiperparámetros

1. Ejecutar la notebook `notebook_example/hyperparameter_search.ipynb`
2. Se entrenan distintos modelos:
- Logistig Regression
- Random Forest
- XGBoost
3. Se selecciona el mejor modelo según:
- ROC-AUC
- F1-score
- Brier Score
4. El modelo se registra en MLFlow como **champion**.

### 4. Usar la API de predicción

```bash
curl -X 'POST' \
  'http://localhost:8800/predict/' \
  -H 'Content-Type: application/json' \
  -d '{
  "features": {
    "Location": "Sydney",
    "MinTemp": 13.4,
    "MaxTemp": 22.9,
    "Rainfall": 0.6,
    "Humidity3pm": 71,
    "Pressure3pm": 1015.3,
    "WindSpeed3pm": 24
  }
}'
```

Respuesta esperada:
```json
{
  "int_output": 1,
  "str_output": "Rain expected tomorrow"
}
```

### 5. Reentrenamiento del modelo (opcional)

1. Ejecutar primero el DAG `process_etl_rain_data` para generar datos nuevos
2. Ejecutar el DAG `retrain_rain_model`
3. El DAG compara el nuevo modelo con el champion y promueve si es mejor

## Apagar los servicios

```bash
docker compose --profile all down
```

Para eliminar toda la infraestructura (datos incluidos):

```bash
docker compose down --rmi all --volumes
```

## Conexión con los buckets (desde local)

Para conectarse a MinIO desde notebooks o scripts locales:

```python
import os
os.environ['AWS_ACCESS_KEY_ID'] = 'minio'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minio123'
os.environ['AWS_ENDPOINT_URL_S3'] = 'http://localhost:9000'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000'
```

## TODOs para el grupo

- [ ] Generar modelo fallback (model.pkl)
- [ ] Agregar más métricas o visualizaciones al notebook de hiperparámetros
- [ ] Considerar usar Optuna en lugar de GridSearchCV para optimización más eficiente
- [ ] Agregar tests unitarios para la API
- [ ] Mejorar documentación de los endpoints de FastAPI
- [ ] Considerar agregar un DAG de monitoreo o drift detection

## Basado en

- [amq2-service-ml](https://github.com/facundolucianna/amq2-service-ml) por Facundo Lucianna
- Branch de ejemplo: [example_implementation](https://github.com/facundolucianna/amq2-service-ml/tree/example_implementation)
