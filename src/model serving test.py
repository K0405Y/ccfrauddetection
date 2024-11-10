# Databricks notebook source
from mlflow.models import validate_serving_input
import warnings 
warnings.filterwarnings("ignore")

model_uri = 'runs:/44e32bbfe1e64329a925d0dc40441b7f/fraud_detection_inference'

# The model is logged with an input example. MLflow converts
# it into the serving payload format for the deployed model endpoint,
# and saves it to 'serving_input_payload.json'
serving_payload = """{
  "inputs": {
    "TRANSACTION_ID": 4781,
    "TX_DATETIME": "2024-10-29 05:57:40",
    "CUSTOMER_ID": 17085,
    "TERMINAL_ID": 139,
    "TX_AMOUNT": 251.25,
    "TX_TIME_SECONDS": 21460,
    "TX_TIME_DAYS": 0
  }
}"""

# Validate the serving payload works on the model
validate_serving_input(model_uri, serving_payload)
