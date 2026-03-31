FROM postgres:13

# Create the mlflow_db database on initialization
COPY mlflow.sql /docker-entrypoint-initdb.d/
