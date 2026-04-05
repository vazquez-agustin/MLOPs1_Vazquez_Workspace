# Proyecto Final - MLOps1: RainTomorrow Prediction Model

### Operaciones de Aprendizaje Automático I - CEIA - FIUBA

**Equipo:**

* Paola Andrea Blanco (a2303)
* Agustín Jesús Vazquez (e2301)
* Facundo Manuel Quiroga (a2305)
* Victor Gabriel Peralta (a2322)

**Profesor:** Facundo Lucianna

---

## Descripción

Implementación productiva de un modelo de machine learning para la predicción de lluvia al día siguiente (`RainTomorrow`), utilizando la infraestructura de **ML Models and something more Inc.**

El modelo fue originalmente desarrollado en la materia **Aprendizaje de Máquina I** sobre el dataset *Rain in Australia*, y en este proyecto se lo lleva a un entorno productivo completo incorporando prácticas de **MLOps**:

* Orquestación de pipelines con Apache Airflow
* Tracking de experimentos y modelos con MLflow
* Data Lake basado en S3 (MinIO)
* Servicio de predicción mediante API REST con FastAPI

## Problema de Negocio

El objetivo es predecir si lloverá al día siguiente en distintas localidades de Australia, a partir de variables meteorológicas actuales.

Este tipo de modelo puede utilizarse para:

* planificación agrícola
* logística y transporte
* toma de decisiones en energía y recursos

---

## Dataset

* **Fuente:** Rain in Australia (Kaggle)
* **Observaciones:** ~145.000
* **Variables:** 23
* **Variable objetivo:** `RainTomorrow` (Yes / No)
* **Desbalance:** ~77% No / 23% Yes

---

## Arquitectura de Servicios

| Servicio | Puerto | Descripción |
| --- | --- | --- |
| Apache Airflow | [localhost:8080](http://localhost:8080) | Orquestación de pipelines ETL, entrenamiento y reentrenamiento |
| MLflow | [localhost:5001](http://localhost:5001) | Tracking de experimentos y registro de modelos |
| MinIO (S3) | [localhost:9001](http://localhost:9001) | Data Lake (almacenamiento de datos y artefactos) |
| FastAPI | [localhost:8800](http://localhost:8800) | REST API para servir predicciones |
| API Docs | [localhost:8800/docs](http://localhost:8800/docs) | Documentación interactiva de la API (Swagger) |
| PostgreSQL | localhost:5432 | Base de datos compartida (Airflow + MLflow) |

---

## Arquitectura del Pipeline

El flujo completo del sistema es el siguiente:

```
Raw → ETL DAG → Entrenamiento Inicial DAG → MLflow (champion) → FastAPI
                                                    ↑
                                         Retrain DAG (periódico)
```

### ETL Pipeline (`process_etl_rain_data`)

* Extracción del dataset desde el directorio local
* Limpieza de valores faltantes
* Encoding de variables categóricas
* Escalado de variables numéricas
* Split train/test
* Almacenamiento en MinIO (`s3://data/final/`)

### Entrenamiento Inicial (`train_initial_model`)

* Carga de datos procesados desde MinIO
* Búsqueda de hiperparámetros con `RandomizedSearchCV` (10 iteraciones × 2 folds = 20 fits)
* Modelo: XGBoost optimizando F1-score (métrica adecuada para datos desbalanceados)
* Registro de parámetros, métricas y modelo en MLflow
* Asignación del alias `champion` al mejor modelo

### Reentrenamiento (`retrain_the_model`)

* Carga de datos frescos desde MinIO
* Reentrenamiento con los hiperparámetros del champion actual
* Comparación del nuevo modelo contra el champion (F1-score)
* Promoción automática si el nuevo modelo supera al champion

### Model Serving

* FastAPI carga el modelo con alias `champion` desde MLflow al iniciar
* Endpoint `POST /predict/` recibe variables meteorológicas y devuelve la predicción
* Documentación interactiva disponible en `/docs`

---

## Estructura del Proyecto

```
.
├── docker-compose.yaml               # Definición de todos los servicios
├── .env                              # Variables de configuración
├── .gitignore
├── README.md
├── CONTRIBUTING.md
│
├── airflow/
│   ├── dags/
│   │   ├── etl_rain_process.py       # DAG: Pipeline ETL
│   │   ├── train_initial_model.py    # DAG: Entrenamiento inicial con búsqueda de hiperparámetros
│   │   └── retrain_rain_model.py     # DAG: Reentrenamiento y comparación con champion
│   ├── secrets/
│   │   ├── variables.yaml            # Variables de Airflow
│   │   └── connections.yaml          # Conexiones de Airflow
│   ├── config/
│   ├── logs/
│   └── plugins/
│
├── dockerfiles/
│   ├── airflow/                      # Imagen custom de Airflow
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── mlflow/                       # Imagen custom de MLflow
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── postgres/                     # Imagen custom de PostgreSQL
│   │   ├── Dockerfile
│   │   └── mlflow.sql
│   └── fastapi/                      # Imagen custom de FastAPI
│       ├── Dockerfile
│       ├── requirements.txt          # Incluye xgboost (requerido para cargar el modelo)
│       ├── app.py                    # Aplicación FastAPI
│       └── files/
│           └── data.json             # Datos de referencia para preprocesamiento
│
└── notebook_example/
    └── hyperparameter_search.ipynb   # Búsqueda exhaustiva de hiperparámetros (referencia)
```

---

## Instalación y Uso

### Requisitos Previos

* [Docker Desktop](https://docs.docker.com/engine/install/) instalado y corriendo
* Al menos 4 GB de RAM disponible para Docker
* Al menos 2 CPUs

> **Linux:** Antes de levantar los servicios, correr:
> ```bash
> echo "AIRFLOW_UID=$(id -u)" >> .env
> ```

### 1. Levantar los servicios

```bash
docker compose --profile all up --build -d
```

Esperar hasta que todos los servicios estén healthy:

```bash
docker ps -a
```

### 2. Ejecutar el pipeline ETL

1. Acceder a Airflow: http://localhost:8080 (usuario: `airflow`, contraseña: `airflow`)
2. Activar y ejecutar el DAG `process_etl_rain_data`
3. Esperar que todos los tasks estén en verde
4. Esto genera los datos procesados en el bucket `s3://data/final/`

### 3. Entrenar el modelo inicial

1. En Airflow, activar y ejecutar el DAG `train_initial_model`
2. El DAG realiza una búsqueda de hiperparámetros con `RandomizedSearchCV` sobre XGBoost
3. Registra el mejor modelo en MLflow como **champion**
4. Duración estimada: 5-10 minutos

### 4. Verificar que FastAPI levantó correctamente

```bash
docker ps -a | grep fastapi
```

Debe aparecer como `Up`. Si todavía figura como `Restarting`, reiniciarlo:

```bash
docker restart fastapi
```

### 5. Usar la API de predicción

```bash
curl -X POST 'http://localhost:8800/predict/' \
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

### 6. Reentrenamiento del modelo (opcional)

1. Ejecutar primero `process_etl_rain_data` para generar datos nuevos
2. Ejecutar el DAG `retrain_the_model`
3. El DAG compara el nuevo modelo contra el champion y lo promueve si es mejor

---

## Apagar los servicios

```bash
docker compose --profile all down
```

Para eliminar toda la infraestructura incluyendo datos y volúmenes:

```bash
docker compose down --rmi all --volumes
```

---

## Conexión con MinIO desde local

Para conectarse a MinIO desde notebooks o scripts locales:

```python
import os
os.environ['AWS_ACCESS_KEY_ID'] = 'minio'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minio123'
os.environ['AWS_ENDPOINT_URL_S3'] = 'http://localhost:9000'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000'
```

---

## Estado del Proyecto

### Implementado ✅

* Pipeline ETL completo orquestado con Airflow
* DAG de entrenamiento inicial con búsqueda de hiperparámetros (RandomizedSearchCV)
* DAG de reentrenamiento con patrón champion/challenger
* Tracking de experimentos y Model Registry con MLflow
* API REST con FastAPI sirviendo predicciones del modelo champion
* Infraestructura completa dockerizada (Airflow, MLflow, MinIO, PostgreSQL, FastAPI)
* `xgboost` agregado como dependencia en la imagen de FastAPI

### Pendiente ⚠️

* Generar modelo fallback (`model.pkl`) para que FastAPI pueda arrancar sin un champion en MLflow
* Agregar tests unitarios para los endpoints de FastAPI
* Mejorar documentación de los endpoints en Swagger (descripciones, ejemplos, códigos de error)
* Agregar un DAG de monitoreo o drift detection para detectar degradación del modelo en producción

---

## Basado en

* [amq2-service-ml](https://github.com/facundolucianna/amq2-service-ml) por Facundo Lucianna
* Branch de ejemplo: [example_implementation](https://github.com/facundolucianna/amq2-service-ml/tree/example_implementation)