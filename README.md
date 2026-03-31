# Proyecto Final - MLOps1: Heart Disease SVM Model

### Operaciones de Aprendizaje Automático I - CEIA - FIUBA

**Equipo:**
- Paola Andrea Blanco (a2303)
- Agustín Jesús Vazquez (e2301)
- Facundo Manuel Quiroga (a2305)
- Victor Gabriel Peralta (a2322)

**Profesor:** Facundo Lucianna

---

## Descripción

Implementación productiva de un modelo SVM para la detección de enfermedad cardíaca, utilizando la infraestructura de **ML Models and something more Inc.**

El modelo fue originalmente desarrollado en la materia Aprendizaje de Máquina sobre el dataset [Heart Disease (UCI ML Repository)](https://archive.ics.uci.edu/dataset/45/heart+disease), y esta implementación lo lleva a un entorno productivo completo con orquestación, tracking de experimentos y servicio REST API.

## Arquitectura de Servicios

| Servicio | Puerto | Descripción |
|---|---|---|
| Apache Airflow | [localhost:8080](http://localhost:8080) | Orquestación de pipelines ETL y reentrenamiento |
| MLflow | [localhost:5001](http://localhost:5001) | Tracking de experimentos y registro de modelos |
| MinIO (S3) | [localhost:9001](http://localhost:9001) | Data Lake (almacenamiento de datos y artefactos) |
| FastAPI | [localhost:8800](http://localhost:8800) | REST API para servir predicciones |
| API Docs | [localhost:8800/docs](http://localhost:8800/docs) | Documentación interactiva de la API (Swagger) |
| PostgreSQL | localhost:5432 | Base de datos compartida (Airflow + MLflow) |

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
│   │   ├── etl_process.py       # DAG: Pipeline ETL (fetch → clean → split → normalize)
│   │   └── retrain_the_model.py # DAG: Reentrenamiento y comparación con champion
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
2. Activar y ejecutar el DAG `process_etl_heart_data`
3. Esto creará los datos procesados en el bucket `s3://data`

### 3. Búsqueda de hiperparámetros

1. Ejecutar la notebook `notebook_example/hyperparameter_search.ipynb`
2. Esto entrenará múltiples modelos SVM, registrará el mejor en MLflow, y lo establecerá como "champion"

### 4. Usar la API de predicción

```bash
curl -X 'POST' \
  'http://localhost:8800/predict/' \
  -H 'Content-Type: application/json' \
  -d '{
  "features": {
    "age": 67,
    "sex": 1,
    "cp": 4,
    "trestbps": 160,
    "chol": 286,
    "fbs": 0,
    "restecg": 2,
    "thalach": 108,
    "exang": 1,
    "oldpeak": 1.5,
    "slope": 2,
    "ca": 3,
    "thal": 3
  }
}'
```

Respuesta esperada:
```json
{
  "int_output": true,
  "str_output": "Heart disease detected"
}
```

### 5. Reentrenamiento del modelo (opcional)

1. Ejecutar primero el DAG `process_etl_heart_data` para generar datos nuevos
2. Ejecutar el DAG `retrain_the_model`
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

- [ ] Generar `files/model.pkl` - modelo SVM por defecto (fallback para la API)
- [ ] Agregar más métricas o visualizaciones al notebook de hiperparámetros
- [ ] Considerar usar Optuna en lugar de GridSearchCV para optimización más eficiente
- [ ] Agregar tests unitarios para la API
- [ ] Mejorar documentación de los endpoints de FastAPI
- [ ] Considerar agregar un DAG de monitoreo o drift detection

## Basado en

- [amq2-service-ml](https://github.com/facundolucianna/amq2-service-ml) por Facundo Lucianna
- Branch de ejemplo: [example_implementation](https://github.com/facundolucianna/amq2-service-ml/tree/example_implementation)
