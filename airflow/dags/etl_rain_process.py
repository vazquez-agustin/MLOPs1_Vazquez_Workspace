"""
### ETL Process for Rain Tomorrow Prediction

This DAG extracts information from the **weatherAUS.csv** dataset
(Rain in Australia — Kaggle).

It preprocesses the data by:
1. Reading raw data from the local CSV file
2. Selecting relevant features and converting the target variable
3. Cleaning duplicates and NaN values
4. Creating dummy variables for categorical features (Location)
5. Splitting into train/test sets (70/30, stratified)
6. Normalizing numerical features with StandardScaler

After preprocessing, the data is saved into a S3 bucket as separate
CSV files for training and testing.

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
    dag_id="process_etl_rain_data",
    description="ETL process for rain prediction data: fetch, clean, feature engineer, split, and normalize.",
    doc_md=markdown_text,
    tags=["ETL", "Rain Prediction"],
    default_args=default_args,
    catchup=False,
)
def process_etl_rain_data():
    """Main DAG definition for the Rain Tomorrow ETL pipeline."""

    @task.virtualenv(
        task_id="obtain_original_data",
        requirements=["awswrangler==3.6.0"],
        system_site_packages=True
    )
    def get_data():
        """
        Load the raw weatherAUS data from the local CSV file
        and save it to S3.

        - Reads dataset/weatherAUS.csv mounted at /opt/airflow/dataset/
        - Selects relevant feature columns and target variable
        - Converts target RainTomorrow: Yes → 1, No → 0
        - Saves raw CSV to s3://data/raw/weather.csv
        """
        import awswrangler as wr
        import pandas as pd
        from airflow.models import Variable

        # Read dataset from local mount
        dataframe = pd.read_csv("/opt/airflow/dataset/weatherAUS.csv")

        target_col = Variable.get("target_col_rain")

        # Select relevant features
        selected_columns = [
            "Location", "MinTemp", "MaxTemp", "Rainfall",
            "Humidity3pm", "Pressure3pm", "WindSpeed3pm",
            target_col
        ]
        dataframe = dataframe[selected_columns]

        # Convert target to binary: Yes → 1, No → 0
        dataframe[target_col] = dataframe[target_col].map({"Yes": 1, "No": 0})

        data_path = "s3://data/raw/weather.csv"
        wr.s3.to_csv(df=dataframe,
                     path=data_path,
                     index=False)

    @task.virtualenv(
        task_id="make_dummies_variables",
        requirements=["awswrangler==3.6.0"],
        system_site_packages=True
    )
    def make_dummies_variables():
        """
        Convert categorical variables into one-hot encoding (dummy variables).

        Steps:
        - Read raw data from S3
        - Clean duplicates and NaN values
        - Apply pd.get_dummies on: Location
        - Save processed data to s3://data/raw/weather_dummies.csv
        - Save dataset metadata (columns, dtypes, categories) to s3://data/data_info/data.json
        - Log datasets to MLflow experiment
        """
        import json
        import datetime
        import boto3
        import botocore.exceptions
        import mlflow

        import awswrangler as wr
        import pandas as pd
        import numpy as np

        from airflow.models import Variable

        data_original_path = "s3://data/raw/weather.csv"
        data_end_path = "s3://data/raw/weather_dummies.csv"
        dataset = wr.s3.read_csv(data_original_path)

        # Clean duplicates
        dataset.drop_duplicates(inplace=True, ignore_index=True)
        # Drop NaN
        dataset.dropna(inplace=True, ignore_index=True)

        # Categorical columns to encode
        categories_list = ["Location"]

        # Create dummy variables
        dataset_with_dummies = pd.get_dummies(
            data=dataset,
            columns=categories_list,
            drop_first=True
        )

        wr.s3.to_csv(df=dataset_with_dummies,
                     path=data_end_path,
                     index=False)

        # Save metadata about the dataset to S3
        client = boto3.client('s3')

        data_dict = {}
        try:
            client.head_object(Bucket='data', Key='data_info/data.json')
            result = client.get_object(Bucket='data', Key='data_info/data.json')
            text = result["Body"].read().decode()
            data_dict = json.loads(text)
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] != "404":
                raise e

        target_col = Variable.get("target_col_rain")
        dataset_log = dataset.drop(columns=target_col)
        dataset_with_dummies_log = dataset_with_dummies.drop(columns=target_col)

        # Build metadata dictionary
        data_dict['columns'] = dataset_log.columns.to_list()
        data_dict['columns_after_dummy'] = dataset_with_dummies_log.columns.to_list()
        data_dict['target_col'] = target_col
        data_dict['categorical_columns'] = categories_list
        data_dict['columns_dtypes'] = {k: str(v) for k, v in dataset_log.dtypes.to_dict().items()}
        data_dict['columns_dtypes_after_dummy'] = {
            k: str(v) for k, v in dataset_with_dummies_log.dtypes.to_dict().items()
        }

        category_dummies_dict = {}
        for category in categories_list:
            category_dummies_dict[category] = sorted(dataset_log[category].unique().tolist())
        data_dict['categories_values_per_categorical'] = category_dummies_dict

        data_dict['date'] = datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S')
        data_string = json.dumps(data_dict, indent=2)

        client.put_object(
            Bucket='data',
            Key='data_info/data.json',
            Body=data_string
        )

        # Log to MLflow
        mlflow.set_tracking_uri('http://mlflow:5000')
        experiment = mlflow.set_experiment("Rain Prediction")

        mlflow.start_run(
            run_name='ETL_run_' + datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S'),
            experiment_id=experiment.experiment_id,
            tags={"experiment": "etl", "dataset": "Rain in Australia"},
            log_system_metrics=True
        )

        mlflow_dataset = mlflow.data.from_pandas(
            dataset,
            source="https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package",
            targets=target_col,
            name="weather_data_complete"
        )
        mlflow_dataset_dummies = mlflow.data.from_pandas(
            dataset_with_dummies,
            source="https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package",
            targets=target_col,
            name="weather_data_complete_with_dummies"
        )
        mlflow.log_input(mlflow_dataset, context="Dataset")
        mlflow.log_input(mlflow_dataset_dummies, context="Dataset")

    @task.virtualenv(
        task_id="split_dataset",
        requirements=["awswrangler==3.6.0",
                       "scikit-learn==1.3.2"],
        system_site_packages=True
    )
    def split_dataset():
        """
        Split the processed dataset into training and test sets.

        - 70/30 stratified split
        - Saves X_train, X_test, y_train, y_test to s3://data/final/
        """
        import awswrangler as wr
        from sklearn.model_selection import train_test_split
        from airflow.models import Variable

        def save_to_csv(df, path):
            wr.s3.to_csv(df=df, path=path, index=False)

        data_original_path = "s3://data/raw/weather_dummies.csv"
        dataset = wr.s3.read_csv(data_original_path)

        test_size = float(Variable.get("test_size_rain"))
        target_col = Variable.get("target_col_rain")

        X = dataset.drop(columns=target_col)
        y = dataset[[target_col]]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y
        )

        save_to_csv(X_train, "s3://data/final/train/weather_X_train.csv")
        save_to_csv(X_test, "s3://data/final/test/weather_X_test.csv")
        save_to_csv(y_train, "s3://data/final/train/weather_y_train.csv")
        save_to_csv(y_test, "s3://data/final/test/weather_y_test.csv")

    @task.virtualenv(
        task_id="normalize_numerical_features",
        requirements=["awswrangler==3.6.0",
                       "scikit-learn==1.3.2",
                       "mlflow==2.10.2"],
        system_site_packages=True
    )
    def normalize_data():
        """
        Standardize numerical columns using StandardScaler.

        - Fit scaler on training data, transform both train and test
        - Save scaler parameters (mean, std) to data.json in S3
        - Log normalization parameters to MLflow
        """
        import json
        import mlflow
        import boto3
        import botocore.exceptions

        import awswrangler as wr
        import pandas as pd

        from sklearn.preprocessing import StandardScaler

        def save_to_csv(df, path):
            wr.s3.to_csv(df=df, path=path, index=False)

        X_train = wr.s3.read_csv("s3://data/final/train/weather_X_train.csv")
        X_test = wr.s3.read_csv("s3://data/final/test/weather_X_test.csv")

        sc_X = StandardScaler(with_mean=True, with_std=True)
        X_train_arr = sc_X.fit_transform(X_train)
        X_test_arr = sc_X.transform(X_test)

        X_train = pd.DataFrame(X_train_arr, columns=X_train.columns)
        X_test = pd.DataFrame(X_test_arr, columns=X_test.columns)

        save_to_csv(X_train, "s3://data/final/train/weather_X_train.csv")
        save_to_csv(X_test, "s3://data/final/test/weather_X_test.csv")

        # Update dataset metadata with scaler info
        client = boto3.client('s3')

        try:
            client.head_object(Bucket='data', Key='data_info/data.json')
            result = client.get_object(Bucket='data', Key='data_info/data.json')
            text = result["Body"].read().decode()
            data_dict = json.loads(text)
        except botocore.exceptions.ClientError as e:
            raise e

        data_dict['standard_scaler_mean'] = sc_X.mean_.tolist()
        data_dict['standard_scaler_std'] = sc_X.scale_.tolist()
        data_string = json.dumps(data_dict, indent=2)

        client.put_object(
            Bucket='data',
            Key='data_info/data.json',
            Body=data_string
        )

        # Log to MLflow
        mlflow.set_tracking_uri('http://mlflow:5000')
        experiment = mlflow.set_experiment("Rain Prediction")

        list_run = mlflow.search_runs([experiment.experiment_id], output_format="list")

        with mlflow.start_run(run_id=list_run[0].info.run_id):
            mlflow.log_param("Train observations", X_train.shape[0])
            mlflow.log_param("Test observations", X_test.shape[0])
            mlflow.log_param("Standard Scaler feature names", sc_X.feature_names_in_.tolist())
            mlflow.log_param("Standard Scaler mean values", sc_X.mean_.tolist())
            mlflow.log_param("Standard Scaler scale values", sc_X.scale_.tolist())

    # Define task dependencies
    get_data() >> make_dummies_variables() >> split_dataset() >> normalize_data()


dag = process_etl_rain_data()
