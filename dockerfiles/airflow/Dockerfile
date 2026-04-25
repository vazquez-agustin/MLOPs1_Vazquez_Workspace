FROM apache/airflow:3.0.2

USER root

# Instalación de dependencias del sistema
RUN apt-get update \
  && apt-get install -y --no-install-recommends git \
  && apt-get autoremove -yqq --purge \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Fix de permisos para uv
#crear el directorio en el home del usuario y asignarle
# permisos correctamente antes de cambiar al usuario airflow.
RUN mkdir -p /home/airflow/.cache/uv && chown -R airflow:root /home/airflow/.cache/uv

USER airflow

RUN pip install --upgrade pip
RUN pip install --no-cache-dir uv apache-airflow[virtualenv]==3.0.2

COPY requirements.txt /requirements.txt
RUN uv pip install --no-cache-dir -r /requirements.txt