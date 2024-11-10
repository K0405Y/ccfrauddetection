import os
import sys
from src.preprocessing.prep import TransactionPreprocessor
import mlflow
import pandas as pd
import numpy as np
import pickle as pkl
from mlflow.pyfunc import PythonModel
from datetime import datetime
import torch
from typing import Dict, Any, Union

class FraudDetectionEnsemble(PythonModel):
    def __init__(self, model_versions):
        self.model_versions = model_versions
        self.xgb_model = None
        self.rf_model = None
        self.nn_model = None
        self.feature_names = None
        self.weights = [0.4, 0.3, 0.3]
        
    def _load_model(self, workspace, model_name, version):
        return mlflow.pyfunc.load_model(
            model_uri=f"models:/{model_name}/{version}"
        )

    def load_context(self, context):
        if "DATABRICKS_RUNTIME_VERSION" in os.environ:
            mlflow.set_tracking_uri("databricks")
        else:
            mlflow.set_tracking_uri("local")
            
        self.xgb_model = self._load_model(
            context, 
            'xgb_model', 
            self.model_versions['xgb_model']
        )
        
        self.rf_model = self._load_model(
            context, 
            'rf_model', 
            self.model_versions['rf_model']
        )
        
        self.nn_model = self._load_model(
            context, 
            'pytorch_model', 
            self.model_versions['pytorch_model']
        )
        
        self.feature_names = [f'feature_{i}' for i in range(1, 38)]

    def _extract_temporal_features(self, dt_str):
        dt = pd.to_datetime(dt_str)
        return {
            'hour_norm': dt.hour / 24,
            'day_norm': dt.dayofweek / 7,
            'month_norm': dt.month / 12,
            'is_weekend': 1 if dt.dayofweek >= 5 else 0,
            'is_night': 1 if (dt.hour >= 23 or dt.hour <= 4) else 0,
            'is_rush_hour': 1 if (8 <= dt.hour <= 10 or 16 <= dt.hour <= 18) else 0
        }

    def _preprocess_input(self, data):
        # Handle nested structure
        if isinstance(data, dict):
            if 'inputs' in data:
                data = data['inputs']
            if isinstance(data, dict):
                data = pd.DataFrame([data])
        
        # Create features DataFrame
        features = pd.DataFrame()
        
        # Extract temporal features
        temp_features = self._extract_temporal_features(data['TX_DATETIME'].iloc[0])
        
        # Customer features
        features['feature_1'] = data['CUSTOMER_ID'].astype(float) / 10000
        features['feature_2'] = data['TX_TIME_SECONDS'].astype(float) / 86400
        features['feature_3'] = data['TX_TIME_DAYS'].astype(float) / 7
        
        # Terminal features
        features['feature_4'] = data['TERMINAL_ID'].astype(float) / 1000
        
        # Amount features
        features['feature_5'] = data['TX_AMOUNT'].astype(float)
        features['feature_6'] = np.log1p(data['TX_AMOUNT'].astype(float))
        
        # Temporal features
        features['feature_7'] = temp_features['hour_norm']
        features['feature_8'] = temp_features['day_norm']
        features['feature_9'] = temp_features['month_norm']
        features['feature_10'] = temp_features['is_weekend']
        features['feature_11'] = temp_features['is_night']
        features['feature_12'] = temp_features['is_rush_hour']
        
        # Fill remaining features with zeros
        for i in range(13, 38):
            features[f'feature_{i}'] = 0.0
            
        return features

    def predict(self, context, data):
        try:
            # Preprocess input data
            X = self._preprocess_input(data)
            
            # Get individual model predictions
            xgb_prob = self.xgb_model.predict(X)
            rf_prob = self.rf_model.predict(X)
            nn_prob = self.nn_model.predict(X)
            
            # Convert predictions to probabilities if needed
            if hasattr(xgb_prob, 'iloc'):
                xgb_prob = float(xgb_prob.iloc[0])
            elif isinstance(xgb_prob, np.ndarray):
                xgb_prob = float(xgb_prob[0])
            
            if hasattr(rf_prob, 'iloc'):
                rf_prob = float(rf_prob.iloc[0])
            elif isinstance(rf_prob, np.ndarray):
                rf_prob = float(rf_prob[0])
            
            if hasattr(nn_prob, 'iloc'):
                nn_prob = float(nn_prob.iloc[0])
            elif isinstance(nn_prob, np.ndarray):
                nn_prob = float(nn_prob[0])
            
            # Calculate ensemble probability
            ensemble_prob = (
                self.weights[0] * xgb_prob +
                self.weights[1] * rf_prob +
                self.weights[2] * nn_prob
            )
            
            # Make prediction
            prediction = "TXN IS FRAUDULENT" if ensemble_prob >= 0.5 else "TXN IS NOT FRAUDULENT"
            
            return [{
                'prediction': prediction,
                'probability': ensemble_prob,
                'model_predictions': {
                    'xgboost': xgb_prob,
                    'random_forest': rf_prob,
                    'neural_network': nn_prob
                },
                'ensemble_weights': {
                    'xgboost': self.weights[0],
                    'random_forest': self.weights[1],
                    'neural_network': self.weights[2]
                }
            }]
            
        except Exception as e:
            raise RuntimeError(f"Prediction error: {str(e)}")


# Example input
example_input = {
    "TRANSACTION_ID": 4781,
    "TX_DATETIME": "2024-10-29 05:57:40",
    "CUSTOMER_ID": 17085,
    "TERMINAL_ID": 139,
    "TX_AMOUNT": 251.25,
    "TX_TIME_SECONDS": 21460,
    "TX_TIME_DAYS": 0
}

# Model versions
model_versions = {
    'xgb_model': '1',
    'rf_model': '1',
    'pytorch_model': '1'
}

# Initialize and register the model
model = FraudDetectionEnsemble(model_versions)

experiment_id = mlflow.set_experiment('/Users/kehinde.awomuti@pwc.com/fraud_detection_inference')
# Register the model
with mlflow.start_run(run_name='inference_engine') as run:
    mlflow.pyfunc.log_model(
        artifact_path="fraud_detection_inference",
        python_model=model,
        registered_model_name="fraud_detection_model",
        input_example=example_input
    )