"""
### Entrenamiento Inicial - Modelo de Predicción de Lluvia

Este DAG realiza el entrenamiento inicial del modelo XGBoost para predecir
si lloverá al día siguiente (RainTomorrow).

Utiliza RandomizedSearchCV con configuración liviana para adaptarse a los
recursos disponibles en el worker de Airflow:
- n_iter=10: prueba solo 10 combinaciones aleatorias
- cv=2: 2-fold cross-validation
- Total: 10 × 2 = 20 fits (vs 540 del GridSearch original)

Esto reduce el tiempo de ejecución a ~3-5 minutos manteniendo
un resultado de calidad razonable para producción.

Debe ejecutarse una sola vez, después de que el DAG de ETL haya procesado
y cargado los datos en el bucket de MinIO (s3://data/).

**Autores:** Paola Blanco, Agustín Vazquez, Facundo Quiroga, Victor Peralta
"""

import datetime

from airflow.decorators import dag, task

markdown_text = __doc__

# Configuración por defecto del DAG:
# - Sin dependencia de ejecuciones anteriores
# - Sin schedule: se ejecuta manualmente una única vez
# - 1 reintento ante falla, esperando 5 minutos
# - Timeout de 15 minutos: con 20 fits debería ser más que suficiente
default_args = {
    'owner': "MLOps1_Grupo_Vazquez",
    'depends_on_past': False,
    'schedule_interval': None,
    'retries': 1,
    'retry_delay': datetime.timedelta(minutes=5),
    'dagrun_timeout': datetime.timedelta(minutes=15),
}


@dag(
    dag_id="train_initial_model",
    description="Búsqueda aleatoria liviana de hiperparámetros y registro del champion inicial para el modelo XGBoost.",
    doc_md=markdown_text,
    tags=["Entrenamiento", "Rain Prediction", "XGBoost", "RandomizedSearchCV"],
    default_args=default_args,
    catchup=False,
)
def train_initial_model():
    """DAG para entrenar el modelo inicial con búsqueda aleatoria liviana y registrarlo como champion."""

    @task.virtualenv(
        task_id="busqueda_hiperparametros",
        requirements=[
            "awswrangler==3.6.0",
            "scikit-learn==1.3.2",
            "mlflow==2.10.2",
            "xgboost==2.0.3",
        ],
        system_site_packages=True,
    )
    def busqueda_hiperparametros():
        """
        Tarea 1: Búsqueda aleatoria liviana de hiperparámetros.

        Configuración reducida para ajustarse a los recursos del worker:
          - n_iter=10: solo 10 combinaciones aleatorias
          - cv=2: 2-fold cross-validation
          - Total: 20 fits (vs 540 del GridSearch original)

        Pasos:
          1. Carga los datos preprocesados desde MinIO (s3://data/final/)
          2. Ejecuta RandomizedSearchCV con 20 fits optimizando F1-score
          3. Evalúa el mejor modelo sobre el conjunto de test
          4. Registra parámetros, métricas y el modelo en MLflow
          5. Retorna el run_id para que la siguiente tarea registre el modelo
        """
        import mlflow
        import awswrangler as wr
        import pandas as pd
        from xgboost import XGBClassifier
        from sklearn.model_selection import RandomizedSearchCV
        from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

        # ----------------------------------------------------------------
        # 1. Carga de datos desde MinIO (Data Lake)
        # Los datos fueron generados por el DAG process_etl_rain_data.
        # Ya están divididos en train/test, limpios y normalizados.
        # ----------------------------------------------------------------
        X_train = wr.s3.read_csv("s3://data/final/train/weather_X_train.csv")
        y_train = wr.s3.read_csv("s3://data/final/train/weather_y_train.csv").values.ravel()
        X_test = wr.s3.read_csv("s3://data/final/test/weather_X_test.csv")
        y_test = wr.s3.read_csv("s3://data/final/test/weather_y_test.csv").values.ravel()

        print(f"Conjunto de entrenamiento: {X_train.shape[0]} muestras, {X_train.shape[1]} features")
        print(f"Conjunto de test: {X_test.shape[0]} muestras, {X_test.shape[1]} features")

        # ----------------------------------------------------------------
        # 2. Configuración de MLflow
        # Se conecta al servidor MLflow interno usando el nombre del
        # contenedor Docker (mlflow:5000), resoluble dentro de la red
        # interna de Docker Compose.
        # ----------------------------------------------------------------
        mlflow.set_tracking_uri('http://mlflow:5000')
        experimento = mlflow.set_experiment("Rain Prediction")

        # ----------------------------------------------------------------
        # 3. Espacio de búsqueda de hiperparámetros
        # Se mantiene el mismo espacio que en la notebook para consistencia.
        # RandomizedSearchCV va a samplear aleatoriamente 10 combinaciones
        # de este espacio, en lugar de probarlas todas.
        #
        # Justificación de cada hiperparámetro:
        # - n_estimators: cantidad de árboles. Más árboles = más capacidad.
        # - max_depth: profundidad del árbol. Controla el overfitting.
        # - learning_rate: paso de aprendizaje. Valores bajos generalizan mejor.
        # - subsample: fracción de datos por árbol. Reduce overfitting.
        # - scale_pos_weight: ajuste para desbalance de clases (~4x más No lluvia).
        # ----------------------------------------------------------------
        espacio_hiperparametros = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0],
            'scale_pos_weight': [1, 4],
        }

        # ----------------------------------------------------------------
        # 4. Entrenamiento con RandomizedSearchCV (configuración liviana)
        # - n_iter=10: prueba solo 10 combinaciones aleatorias del espacio
        # - cv=2: 2-fold cross-validation (mínimo estadísticamente válido)
        # - Total: 10 × 2 = 20 fits
        #
        # Se optimiza F1-score porque el dataset está desbalanceado
        # (77% No lluvia / 23% Lluvia). La accuracy sería engañosa.
        # random_state=42 garantiza reproducibilidad en la selección
        # de combinaciones y en el entrenamiento.
        # ----------------------------------------------------------------
        modelo_base = XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',  # métrica interna de XGBoost
            random_state=42,        # semilla para reproducibilidad
            n_jobs=-1,              # usa todos los núcleos disponibles
        )

        busqueda = RandomizedSearchCV(
            estimator=modelo_base,
            param_distributions=espacio_hiperparametros,
            n_iter=10,          # solo 10 combinaciones aleatorias
            scoring='f1',       # optimizar F1-score
            cv=2,               # 2-fold cross-validation
            n_jobs=-1,
            random_state=42,    # reproducibilidad
            verbose=1,
        )

        print(f"Iniciando búsqueda: {busqueda.n_iter} combinaciones × {busqueda.cv} folds = {busqueda.n_iter * busqueda.cv} fits")
        busqueda.fit(X_train, y_train)
        mejor_modelo = busqueda.best_estimator_

        print(f"Mejores hiperparámetros: {busqueda.best_params_}")
        print(f"Mejor F1 en CV: {busqueda.best_score_:.4f}")

        # ----------------------------------------------------------------
        # 5. Evaluación sobre el conjunto de test
        # El test set nunca fue visto durante la búsqueda, por lo que
        # estas métricas son una estimación honesta del rendimiento
        # esperado en producción.
        # ----------------------------------------------------------------
        y_pred = mejor_modelo.predict(X_test)
        f1_test = f1_score(y_test, y_pred)
        accuracy_test = accuracy_score(y_test, y_pred)
        precision_test = precision_score(y_test, y_pred)
        recall_test = recall_score(y_test, y_pred)

        print(f"F1 en test:        {f1_test:.4f}")
        print(f"Accuracy en test:  {accuracy_test:.4f}")
        print(f"Precision en test: {precision_test:.4f}")
        print(f"Recall en test:    {recall_test:.4f}")

        # ----------------------------------------------------------------
        # 6. Registro en MLflow
        # Se loguean parámetros, métricas y el modelo para trazabilidad.
        # El run_id se retorna para pasárselo a la siguiente tarea
        # mediante XCom.
        # ----------------------------------------------------------------
        with mlflow.start_run(
            run_name='entrenamiento_inicial_' + pd.Timestamp.now().strftime('%Y%m%d_%H%M%S'),
            experiment_id=experimento.experiment_id,
            tags={
                'experimento': 'entrenamiento_inicial',
                'modelo': 'XGBoost',
                'metodo_busqueda': 'RandomizedSearchCV',
                'dataset': 'Rain in Australia',
            },
        ) as run:
            mlflow.log_params(busqueda.best_params_)
            mlflow.log_param('cv_folds', 2)
            mlflow.log_param('n_iter', 10)

            mlflow.log_metric('f1_score', f1_test)
            mlflow.log_metric('accuracy', accuracy_test)
            mlflow.log_metric('precision', precision_test)
            mlflow.log_metric('recall', recall_test)
            mlflow.log_metric('cv_best_f1', busqueda.best_score_)

            # Serializar y subir el modelo a MLflow/MinIO
            mlflow.sklearn.log_model(mejor_modelo, "xgboost_rain_prediction")

            return run.info.run_id

    @task.virtualenv(
        task_id="registrar_champion",
        requirements=[
            "mlflow==2.10.2",
        ],
        system_site_packages=True,
    )
    def registrar_champion(nuevo_run_id: str):
        """
        Tarea 2: Registrar el mejor modelo como champion en el MLflow Model Registry.

        Esta tarea siempre promueve el modelo sin comparar porque es el
        entrenamiento inicial: no existe un champion previo.

        Pasos:
          1. Registra el modelo en el Model Registry como 'rain_prediction_model_prod'
          2. Asigna el alias 'champion' a esta versión
          3. FastAPI puede cargar el modelo desde este momento usando el alias
        """
        import mlflow

        mlflow.set_tracking_uri('http://mlflow:5000')
        cliente = mlflow.MlflowClient()

        nombre_modelo = "rain_prediction_model_prod"

        # ----------------------------------------------------------------
        # Registrar en el Model Registry
        # Centraliza las versiones del modelo de producción y permite
        # cargarlos por alias sin conocer la ruta exacta en MinIO.
        # ----------------------------------------------------------------
        uri_modelo = f"runs:/{nuevo_run_id}/xgboost_rain_prediction"
        version = mlflow.register_model(uri_modelo, nombre_modelo)
        print(f"Modelo registrado: {nombre_modelo}, versión: {version.version}")

        # ----------------------------------------------------------------
        # Asignar alias 'champion'
        # FastAPI y el DAG de reentrenamiento usan este alias para
        # identificar el modelo activo en producción.
        # ----------------------------------------------------------------
        cliente.set_registered_model_alias(nombre_modelo, "champion", version.version)
        print(f"Versión {version.version} asignada como champion!")

        metricas = cliente.get_run(nuevo_run_id).data.metrics
        print(f"F1 del champion:       {metricas.get('f1_score', 0):.4f}")
        print(f"Accuracy del champion: {metricas.get('accuracy', 0):.4f}")
        print(f"F1 en CV del champion: {metricas.get('cv_best_f1', 0):.4f}")

    # ----------------------------------------------------------------
    # Flujo del DAG:
    #   busqueda_hiperparametros → registrar_champion
    #
    # El run_id se pasa entre tareas via XCom.
    # ----------------------------------------------------------------
    mlflow_run_id = busqueda_hiperparametros()
    registrar_champion(mlflow_run_id)


dag = train_initial_model()