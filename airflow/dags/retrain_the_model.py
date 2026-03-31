"""
### Retrain Heart Disease Model

This DAG retrains the Heart Disease SVM model using fresh data from the S3 bucket.
It compares the new model against the current champion model and promotes the new
model if it achieves a better F1 score.

**Authors:** Paola Blanco, Agustín Vazquez, Facundo Quiroga, Victor Peralta
"""

import datetime

from airflow.decorators import dag, task

markdown_text = __doc__

default_args = {
    'owner': "MLOps1_Grupo_Vazquez",
    'depends_on_past': False,
    'schedule_interval': None,
    'retries': 1,
    'retry_delay': datetime.timedelta(minutes=5),
    'dagrun_timeout': datetime.timedelta(minutes=15)
}


@dag(
    dag_id="retrain_the_model",
    description="Retrain the Heart Disease SVM model and compare with the champion.",
    doc_md=markdown_text,
    tags=["Retrain", "Heart Disease", "SVM"],
    default_args=default_args,
    catchup=False,
)
def retrain_the_model():
    """DAG to retrain the model and compare against the champion."""

    @task.virtualenv(
        task_id="train_new_model",
        requirements=[
            "awswrangler==3.6.0",
            "scikit-learn==1.3.2",
            "mlflow==2.10.2",
            "matplotlib==3.8.2",
            "seaborn==0.13.1",
        ],
        system_site_packages=True,
    )
    def train_new_model():
        """
        Train a new SVM model on the latest data from S3.

        - Loads train/test data from s3://data/final/
        - Trains an SVM with the best hyperparameters from the champion model
        - Logs metrics and model to MLflow
        - Returns the new run_id

        TODO: Complete the SVM training logic.
              Use the hyperparameters from the champion model's MLflow registry
              or define new hyperparameter ranges for GridSearchCV.
        """
        import mlflow
        import awswrangler as wr
        import pandas as pd
        import numpy as np
        from sklearn.svm import SVC
        from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

        # Load data from S3
        X_train = wr.s3.read_csv("s3://data/final/train/heart_X_train.csv")
        y_train = wr.s3.read_csv("s3://data/final/train/heart_y_train.csv")
        X_test = wr.s3.read_csv("s3://data/final/test/heart_X_test.csv")
        y_test = wr.s3.read_csv("s3://data/final/test/heart_y_test.csv")

        y_train = y_train.values.ravel()
        y_test = y_test.values.ravel()

        # Setup MLflow
        mlflow.set_tracking_uri('http://mlflow:5000')
        experiment = mlflow.set_experiment("Heart Disease")

        with mlflow.start_run(
            run_name='retrain_' + pd.Timestamp.now().strftime('%Y%m%d_%H%M%S'),
            experiment_id=experiment.experiment_id,
            tags={"experiment": "retrain", "model": "SVM"},
        ) as run:
            # TODO: Load champion hyperparameters or define new ones
            # For now, use a basic SVM configuration
            model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
            model.fit(X_train, y_train)

            # Evaluate
            y_pred = model.predict(X_test)
            f1 = f1_score(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)

            # Log metrics
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)

            # Log model
            mlflow.sklearn.log_model(model, "svm_heart_disease")

            return run.info.run_id

    @task.virtualenv(
        task_id="compare_and_promote",
        requirements=[
            "mlflow==2.10.2",
            "scikit-learn==1.3.2",
        ],
        system_site_packages=True,
    )
    def compare_and_promote(new_run_id: str):
        """
        Compare the new model with the current champion.

        - Loads both models' F1 scores from MLflow
        - If the new model is better, promotes it as the new champion
        - Otherwise, keeps the existing champion

        TODO: Complete the comparison and promotion logic.
        """
        import mlflow

        mlflow.set_tracking_uri('http://mlflow:5000')
        client = mlflow.MlflowClient()

        model_name = "heart_disease_model_prod"

        # Get new model metrics
        new_run = client.get_run(new_run_id)
        new_f1 = new_run.data.metrics.get("f1_score", 0)

        # Try to get the champion model's F1 score
        try:
            champion_data = client.get_model_version_by_alias(model_name, "champion")
            champion_run = client.get_run(champion_data.run_id)
            champion_f1 = champion_run.data.metrics.get("f1_score", 0)
        except Exception:
            # No champion exists yet, any model is better
            champion_f1 = 0

        # Compare and promote if new model is better
        if new_f1 > champion_f1:
            # Register the new model
            model_uri = f"runs:/{new_run_id}/svm_heart_disease"
            mv = mlflow.register_model(model_uri, model_name)

            # Set as champion
            client.set_registered_model_alias(model_name, "champion", mv.version)
            print(f"New champion! F1: {new_f1:.4f} > {champion_f1:.4f}")
        else:
            print(f"Champion retained. F1: {champion_f1:.4f} >= {new_f1:.4f}")

    # Define task dependencies
    new_run_id = train_new_model()
    compare_and_promote(new_run_id)


dag = retrain_the_model()
