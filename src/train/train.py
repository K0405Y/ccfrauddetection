import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)
from src.preprocessing.prep import TransactionPreprocessor
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report,roc_auc_score
import xgboost as xgb
import torch
import torch.nn as nn 
import mlflow
import mlflow.pytorch
import mlflow.sklearn
import shap
from captum.attr import IntegratedGradients, DeepLift
import lime
import lime.lime_tabular

class FraudDetectionNN(nn.Module):
    def __init__(self, input_dim):
        super(FraudDetectionNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)
    
# Custom inference logic for real-time predictions
class FraudDetectionInference:
    def __init__(self, ensemble_model, scaler, feature_cols):
        self.model = ensemble_model
        self.scaler = scaler
        self.feature_cols = feature_cols
    
    def preprocess(self, transaction_data):
        # Extract relevant features
        features = pd.DataFrame([transaction_data])
        features = features[self.feature_cols]
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        return pd.DataFrame(features_scaled, columns=self.feature_cols)
    
    def predict_transaction(self, transaction_data):
        # Preprocess the transaction
        processed_data = self.preprocess(transaction_data)
        
        # Get model prediction
        fraud_probability = self.model.predict_proba(processed_data)[0]
        
        # Add custom business logic
        result = {
            'fraud_probability': float(fraud_probability),
            'is_fraud': bool(fraud_probability >= 0.5),
            'confidence': 'high' if abs(fraud_probability - 0.5) > 0.3 else 'low',
            'timestamp': pd.Timestamp.now()
        }
        
        return result

class ModelInterpretability:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        self.shap_values = None
        
    def compute_shap_values(self, X, sample_size=100):
        """Compute SHAP values for feature importance"""
        if isinstance(self.model, xgb.XGBClassifier):
            explainer = shap.TreeExplainer(self.model)
        else:
            # For neural network, use KernelExplainer
            sample_data = shap.sample(X, sample_size)
            explainer = shap.KernelExplainer(self.model.predict_proba, sample_data)
        
        self.shap_values = explainer.shap_values(X)
        return self.shap_values
    
    def get_feature_importance(self, X):
        """Get global feature importance"""
        importance_dict = {}
        
        if isinstance(self.model, (RandomForestClassifier, xgb.XGBClassifier)):
            # For tree-based models
            importances = self.model.feature_importances_
            for name, importance in zip(self.feature_names, importances):
                importance_dict[name] = float(importance)
                
        elif isinstance(self.model, FraudDetectionNN):
            # For neural network using integrated gradients
            ig = IntegratedGradients(self.model)
            attributions = ig.attribute(X, target=1)
            importances = torch.mean(torch.abs(attributions), dim=0)
            
            for name, importance in zip(self.feature_names, importances):
                importance_dict[name] = float(importance)
        
        return importance_dict
    
    def explain_prediction(self, instance, num_features=10):
        """Explain a single prediction"""
        if self.shap_values is None:
            self.compute_shap_values(instance.reshape(1, -1))
        
        explanation = {
            'feature_contributions': {},
            'top_features': []
        }
        
        # Get feature contributions
        for i, name in enumerate(self.feature_names):
            explanation['feature_contributions'][name] = float(self.shap_values[0][i])
        
        # Get top contributing features
        sorted_features = sorted(
            explanation['feature_contributions'].items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        explanation['top_features'] = sorted_features[:num_features]
        
        return explanation

class CrossValidationManager:
    def __init__(self, n_splits=5, random_state=42):
        self.n_splits = n_splits
        self.random_state = random_state
        self.cv_results = {}
        
    def perform_cv(self, model, feature_groups, labels):
        """Perform cross-validation"""
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        # Initialize metrics storage
        metrics = {
            'auc_scores': [],
            'precision_scores': [],
            'recall_scores': [],
            'f1_scores': []
        }
        
        # Feature importance across folds
        feature_importance_folds = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(
            next(iter(feature_groups.values())), labels
        )):
            # Split data for this fold
            fold_feature_groups = {
                name: features[train_idx] 
                for name, features in feature_groups.items()
            }
            fold_val_feature_groups = {
                name: features[val_idx]
                for name, features in feature_groups.items()
            }
            fold_labels = labels[train_idx]
            fold_val_labels = labels[val_idx]
            
            # Train model
            model.train(
                fold_feature_groups,
                fold_labels,
                fold_val_feature_groups,
                fold_val_labels
            )
            
            # Get predictions
            val_probs = model.predict_proba(fold_val_feature_groups)
            val_preds = (val_probs >= 0.5).astype(int)
            
            # Calculate metrics
            metrics['auc_scores'].append(
                roc_auc_score(fold_val_labels, val_probs)
            )
            report = classification_report(fold_val_labels, val_preds, output_dict=True)
            metrics['precision_scores'].append(report['1']['precision'])
            metrics['recall_scores'].append(report['1']['recall'])
            metrics['f1_scores'].append(report['1']['f1-score'])
            
            # Get feature importance for this fold
            fold_importance = model.get_feature_importance()
            feature_importance_folds.append(fold_importance)
            
            # Log to MLflow
            with mlflow.start_run(nested=True):
                mlflow.log_metrics({
                    f'fold_{fold}_auc': metrics['auc_scores'][-1],
                    f'fold_{fold}_precision': metrics['precision_scores'][-1],
                    f'fold_{fold}_recall': metrics['recall_scores'][-1],
                    f'fold_{fold}_f1': metrics['f1_scores'][-1]
                })
        
        # Calculate aggregate metrics
        self.cv_results = {
            'mean_auc': np.mean(metrics['auc_scores']),
            'std_auc': np.std(metrics['auc_scores']),
            'mean_precision': np.mean(metrics['precision_scores']),
            'mean_recall': np.mean(metrics['recall_scores']),
            'mean_f1': np.mean(metrics['f1_scores']),
            'feature_importance': self._aggregate_feature_importance(feature_importance_folds)
        }
        
        return self.cv_results
    
    def _aggregate_feature_importance(self, importance_folds):
        """Aggregate feature importance across folds"""
        all_features = set()
        for fold_importance in importance_folds:
            all_features.update(fold_importance.keys())
        
        aggregated = {}
        for feature in all_features:
            values = [fold.get(feature, 0) for fold in importance_folds]
            aggregated[feature] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values))
            }
        
        return aggregated

class FraudDetectionEnsemble:
    def __init__(self, input_dim, weights=[0.4, 0.3, 0.3]):
        self.xgb_model = xgb.XGBClassifier(
            scale_pos_weight=10,  # Handling imbalance
            max_depth=5,
            learning_rate=0.1,
            n_estimators=100
        )
        
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            max_depth=10,
            random_state=42
        )
        
        self.nn_model = FraudDetectionNN(input_dim)
        self.weights = weights
        
    def train(self, X_train, y_train, X_val, y_val):
        # Start MLflow run
        with mlflow.start_run(run_name="fraud_detection_ensemble"):
            # Train XGBoost
            self.xgb_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='auc',
                early_stopping_rounds=10
            )
            mlflow.sklearn.log_model(self.xgb_model, "xgboost_model")
            
            # Train Random Forest
            self.rf_model.fit(X_train, y_train)
            mlflow.sklearn.log_model(self.rf_model, "random_forest_model")
            
            # Train Neural Network
            X_train_tensor = torch.FloatTensor(X_train.values)
            y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1)
            
            criterion = nn.BCELoss()
            optimizer = torch.optim.Adam(self.nn_model.parameters(), lr=0.001)
            
            # Neural Network training loop
            self.nn_model.train()
            for epoch in range(10):
                optimizer.zero_grad()
                outputs = self.nn_model(X_train_tensor)
                loss = criterion(outputs, y_train_tensor)
                loss.backward()
                optimizer.step()
                
                mlflow.log_metric("nn_loss", loss.item(), step=epoch)
            
            mlflow.pytorch.log_model(self.nn_model, "pytorch_model")
    
    def predict_proba(self, X):
        # Get predictions from each model
        xgb_pred = self.xgb_model.predict_proba(X)[:, 1]
        rf_pred = self.rf_model.predict_proba(X)[:, 1]
        
        # Neural Network prediction
        self.nn_model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X.values)
            nn_pred = self.nn_model(X_tensor).numpy().flatten()
        
        # Weighted ensemble prediction
        ensemble_pred = (
            self.weights[0] * xgb_pred +
            self.weights[1] * rf_pred +
            self.weights[2] * nn_pred
        )
        
        return ensemble_pred
    
    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def get_feature_importance(self):
        """Get feature importance from all models"""
        importance_dict = {}
        
        # XGBoost importance
        xgb_importance = dict(zip(
            self.feature_names,
            self.xgb_model.feature_importances_
        ))
        
        # Random Forest importance
        rf_importance = dict(zip(
            self.feature_names,
            self.rf_model.feature_importances_
        ))
        
        # Neural Network importance (using integrated gradients)
        ig = IntegratedGradients(self.nn_model)
        nn_importance = {}
        
        # Combine importances with weights
        for feature in self.feature_names:
            importance_dict[feature] = (
                self.weights[0] * xgb_importance[feature] +
                self.weights[1] * rf_importance[feature] +
                self.weights[2] * nn_importance.get(feature, 0)
            )
        
        return importance_dict

class FraudDetectionInference:
    def __init__(self, ensemble_model, preprocessor):
        self.model = ensemble_model
        self.preprocessor = preprocessor
        self.interpreter = ModelInterpretability(
            model=ensemble_model,
            feature_names=self.get_feature_names()
        )
    
    def get_feature_names(self):
        """Get list of all feature names"""
        feature_names = []
        for group, features in self.preprocessor.feature_groups.items():
            feature_names.extend([f"{group}_{feature}" for feature in features])
        return feature_names
    
    def predict_transaction(self, transaction_data):
        # Process transaction
        processed_features = self.preprocessor.transform(
            pd.DataFrame([transaction_data]), training=False
        )
        
        # Get prediction
        fraud_probability = self.model.predict_proba(processed_features)[0]
        
        # Get prediction explanation
        explanation = self.interpreter.explain_prediction(processed_features)
        
        # Prepare result
        result = {
            'fraud_probability': float(fraud_probability),
            'is_fraud': bool(fraud_probability >= 0.5),
            'confidence': 'high' if abs(fraud_probability - 0.5) > 0.3 else 'low',
            'timestamp': pd.Timestamp.now(),
            'explanation': {
                'top_contributing_features': explanation['top_features'],
                'feature_contributions': explanation['feature_contributions']
            }
        }
        
        return result

def train_fraud_detection_system(raw_data):
    # Initialize components
    preprocessor = TransactionPreprocessor()
    cv_manager = CrossValidationManager(n_splits=5)
    
    # Preprocess data
    feature_groups, labels = preprocessor.transform(raw_data, training=True)
    
    # Get feature dimensions
    feature_dims = {
        group: features.shape[1] 
        for group, features in feature_groups.items()
    }
    
    # Initialize ensemble
    ensemble = FraudDetectionEnsemble(feature_dims)
    
    # Perform cross-validation
    cv_results = cv_manager.perform_cv(ensemble, feature_groups, labels)
    
    # Train final model on full dataset
    ensemble.train(feature_groups, labels)
    
    # Log final results to MLflow
    with mlflow.start_run():
        mlflow.log_metrics({
            'final_cv_mean_auc': cv_results['mean_auc'],
            'final_cv_mean_f1': cv_results['mean_f1']
        })
        
        # Log feature importance
        mlflow.log_dict(
            cv_results['feature_importance'],
            'feature_importance.json'
        )
    
    return preprocessor, ensemble, cv_results