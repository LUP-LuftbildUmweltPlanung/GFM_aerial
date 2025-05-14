# mlflow_config.py
import os
import mlflow

# config file from minio
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://74.63.3.44:9000"
os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"
os.environ["AWS_ACCESS_KEY_ID"] = "XeAMQQjZY2pTcXWfxh4H"
os.environ["AWS_SECRET_ACCESS_KEY"] = "wyJ30G38aC2UcyaFjVj2dmXs1bITYkJBcx0FtljZ"
mlflow.set_tracking_uri("http://74.63.3.44:5000")

# Authentication (required if --app-name=basic-auth is enabled)
os.environ["MLFLOW_TRACKING_USERNAME"] = "admin"       # or your custom username
os.environ["MLFLOW_TRACKING_PASSWORD"] = "LUP"    # or your custom password