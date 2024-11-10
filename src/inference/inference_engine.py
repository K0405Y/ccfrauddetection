import os
import mlflow
import pandas as pd
import numpy as np
import pickle as pkl
from mlflow.pyfunc import PythonModel
from datetime import datetime
import logging
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FraudDetectionEnsemble(PythonModel):
    def __init__(self, model_versions):
        """
        Initialize the Fraud Detection Ensemble model
        
        Args:
            model_versions (dict): Dictionary containing model names and versions
                                 e.g. {'xgb_model': '1', 'rf_model': '1', 'pytorch_model': '1'}
        """
        self.model_versions = model_versions
        self.xgb_model = None
        self.rf_model = None
        self.nn_model = None
        self.preprocessor = None
        self.feature_groups = None
        self.weights = [0.4, 0.3, 0.3]  # XGBoost, RandomForest, Neural Network weights
        
    def _load_model(self, workspace, model_name, version):
        """Helper function to load individual models"""
        try:
            model = mlflow.pyfunc.load_model(
                model_uri=f"models:/{model_name}/{version}"
            )
            logger.info(f"Successfully loaded {model_name} version {version}")
            return model
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {str(e)}")
            raise RuntimeError(f"Failed to load {model_name}: {str(e)}")

    def load_context(self, context):
        """
        Load all required models and preprocessing components
        """
        # Set MLflow tracking URI
        if "DATABRICKS_RUNTIME_VERSION" in os.environ:
            mlflow.set_tracking_uri("databricks")
        else:
            mlflow.set_tracking_uri("local")  # or your specific URI
            
        try:
            # Load the individual models
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
            
            # Load feature groups and preprocessor
            try:
                with open('/path/to/feature_groups.pkl', 'rb') as file:
                    self.feature_groups = pkl.load(file)
                logger.info("Successfully loaded feature groups")
            except Exception as e:
                logger.error(f"Failed to load feature groups: {str(e)}")
                raise RuntimeError(f"Failed to load feature groups: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error in load_context: {str(e)}")
            raise RuntimeError(f"Error in load_context: {str(e)}")

    def _preprocess_input(self, input_data):
        """
        Preprocess the input data to match model requirements
        """
        try:
            # Convert input dictionary to DataFrame
            if isinstance(input_data, dict):
                input_data = pd.DataFrame([input_data])
            
            # Initialize preprocessor
            preprocessor = TransactionPreprocessor()
            
            # Transform the input data
            feature_groups = preprocessor.transform(input_data, training=False)
            
            # Concatenate all feature groups
            X = np.concatenate([
                feature_groups[group] 
                for group in sorted(feature_groups.keys())
            ], axis=1)
            
            return X
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            raise RuntimeError(f"Error in preprocessing: {str(e)}")

    def predict(self, context, input_data):
        """
        Make predictions using the ensemble of models
        """
        try:
            # Preprocess input data
            X = self._preprocess_input(input_data)
            
            # Get predictions from each model
            xgb_pred = self.xgb_model.predict_proba(X)[:, 1]
            rf_pred = self.rf_model.predict_proba(X)[:, 1]
            
            # For PyTorch model, need to convert to tensor
            X_tensor = torch.FloatTensor(X)
            with torch.no_grad():
                nn_pred = self.nn_model(X_tensor).numpy()
            
            # Calculate weighted average
            ensemble_pred = (
                self.weights[0] * xgb_pred +
                self.weights[1] * rf_pred +
                self.weights[2] * nn_pred.flatten()
            )
            
            # Make final prediction
            threshold = 0.5
            final_predictions = []
            
            for pred in ensemble_pred:
                if pred >= threshold:
                    final_predictions.append("TXN IS FRAUDULENT")
                else:
                    final_predictions.append("TXN IS NOT FRAUDULENT")
            
            # Add prediction probabilities
            results = []
            for pred, prob in zip(final_predictions, ensemble_pred):
                results.append({
                    'prediction': pred,
                    'probability': float(prob),
                    'model_predictions': {
                        'xgboost': float(xgb_pred[0]),
                        'random_forest': float(rf_pred[0]),
                        'neural_network': float(nn_pred[0])
                    }
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise RuntimeError(f"Error in prediction: {str(e)}")

# Example usage:
if __name__ == "__main__":
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
    
    # Register the model
    mlflow.pyfunc.log_model(
        artifact_path="fraud_detection_ensemble",
        python_model=model,
        registered_model_name="fraud_detection_ensemble",
        input_example=example_input
    )

# class ModelInterpretability:
#     def __init__(self, model, feature_names):
#         self.model = model
#         self.feature_names = feature_names
#         self.shap_values = None
        
#     def compute_shap_values(self, X, sample_size=100):
#         """Compute SHAP values for feature importance"""
#         if isinstance(self.model, xgb.XGBClassifier):
#             explainer = shap.TreeExplainer(self.model)
#         else:
#             # For neural network, use KernelExplainer
#             sample_data = shap.sample(X, sample_size)
#             explainer = shap.KernelExplainer(self.model.predict_proba, sample_data)
        
#         self.shap_values = explainer.shap_values(X)
#         return self.shap_values
    
#     def get_feature_importance(self, X):
#         """Get global feature importance"""
#         importance_dict = {}
        
#         if isinstance(self.model, (RandomForestClassifier, xgb.XGBClassifier)):
#             # For tree-based models
#             importances = self.model.feature_importances_
#             for name, importance in zip(self.feature_names, importances):
#                 importance_dict[name] = float(importance)
                
#         elif isinstance(self.model, FraudDetectionNN):
#             # For neural network using integrated gradients
#             ig = IntegratedGradients(self.model)
#             attributions = ig.attribute(X, target=1)
#             importances = torch.mean(torch.abs(attributions), dim=0)
            
#             for name, importance in zip(self.feature_names, importances):
#                 importance_dict[name] = float(importance)
        
#         return importance_dict
    
#     def explain_prediction(self, instance, num_features=10):
#         """Explain a single prediction"""
#         if self.shap_values is None:
#             self.compute_shap_values(instance.reshape(1, -1))
        
#         explanation = {
#             'feature_contributions': {},
#             'top_features': []
#         }
        
#         # Get feature contributions
#         for i, name in enumerate(self.feature_names):
#             explanation['feature_contributions'][name] = float(self.shap_values[0][i])
        
#         # Get top contributing features
#         sorted_features = sorted(
#             explanation['feature_contributions'].items(),
#             key=lambda x: abs(x[1]),
#             reverse=True
#         )
#         explanation['top_features'] = sorted_features[:num_features]
        
#         return explanation
