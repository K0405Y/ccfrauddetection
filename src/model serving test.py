# Databricks notebook source
import mlflow
from mlflow.models import validate_serving_input
import warnings 
warnings.filterwarnings("ignore")
from databricks.sdk import WorkspaceClient
mlflow.set_tracking_uri("databricks")

model_uri = 'runs:/44e32bbfe1e64329a925d0dc40441b7f/fraud_detection_inference'

# The model is logged with an input example. MLflow converts
# it into the serving payload format for the deployed model endpoint,
# and saves it to 'serving_input_payload.json'
serving_payload = """{
  "inputs": {
    "TRANSACTION_ID": 53046,
    "TX_DATETIME": "2024-11-05 00:31:30",
    "CUSTOMER_ID": 158,
    "TERMINAL_ID": 8,
    "TX_AMOUNT": 77.62,
    "TX_TIME_SECONDS": 606690,
    "TX_TIME_DAYS": 7
  }
}"""
# Validate the serving payload works on the model
validate_serving_input(model_uri, serving_payload)
