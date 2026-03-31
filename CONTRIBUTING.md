# Contribución al Proyecto Final - MLOps1

## Equipo

| Nombre | ID |
|---|---|
| Paola Andrea Blanco | a2303 |
| Agustín Jesús Vazquez | e2301 |
| Facundo Manuel Quiroga | a2305 |
| Victor Gabriel Peralta | a2322 |

## Flujo de trabajo

1. **Crear un branch** desde `main` para cada feature o fix:
   ```bash
   git checkout -b feature/nombre-descriptivo
   ```

2. **Commits claros y descriptivos** en español o inglés, con prefijos:
   - `feat:` nueva funcionalidad
   - `fix:` corrección de errores
   - `docs:` documentación
   - `refactor:` refactorización sin cambio funcional
   - `chore:` tareas de mantenimiento (docker, configs)

3. **Pull Request** hacia `main` con revisión de al menos un compañero.

4. **No pushear directo a `main`.**

## Estructura del proyecto

Respetar la estructura de carpetas existente:

- `Proyecto_Final/airflow/dags/` — DAGs de Airflow
- `Proyecto_Final/dockerfiles/` — Dockerfiles e imágenes custom
- `Proyecto_Final/notebook_example/` — Notebooks de experimentación
- `Proyecto_Final/airflow/secrets/` — Variables y conexiones de Airflow

## Buenas prácticas

- Seguir PEP 8 para código Python.
- Documentar funciones con docstrings.
- No subir credenciales ni archivos sensibles (usar `.env` y `.gitignore`).
- No commitear `airflow/logs/`, `__pycache__/`, ni archivos `.pyc`.
- Probar localmente con `docker compose --profile all up` antes de hacer PR.

## Levantar el entorno de desarrollo

```bash
docker compose --profile all up
```

Servicios disponibles tras el inicio:

| Servicio | URL |
|---|---|
| Airflow | http://localhost:8080 |
| MLflow | http://localhost:5001 |
| MinIO | http://localhost:9001 |
| API | http://localhost:8800 |
| API Docs | http://localhost:8800/docs |
