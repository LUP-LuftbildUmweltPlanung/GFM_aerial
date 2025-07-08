# mlflow_config.py
import os
import mlflow

log_to_mlflow = False # set to True for logging to MLflow

# config file from minio
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "URL" # your custom URL
os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"
os.environ["AWS_ACCESS_KEY_ID"] = "ACCESS_KEY" # your custom access key
os.environ["AWS_SECRET_ACCESS_KEY"] = "SECRET_ACCESS_KEY" # your custom secret access key
mlflow.set_tracking_uri("URL") # your custom URL

# Authentication (required if --app-name=basic-auth is enabled)
os.environ["MLFLOW_TRACKING_USERNAME"] = ""       # or your custom username, if you want to restrict access
os.environ["MLFLOW_TRACKING_PASSWORD"] = ""    # or your custom password, if you want to restrict access