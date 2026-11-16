# Serveur MLflow avec les pilotes PostgreSQL (backend store) et S3/MinIO (artifact store)
FROM python:3.12-slim
RUN pip install --no-cache-dir mlflow==3.18.0 psycopg2-binary boto3
EXPOSE 5000
