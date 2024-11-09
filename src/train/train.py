import sys
import os
from src.preprocessing.prep import TransactionPreprocessor
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import  train_test_split
from sklearn.metrics import roc_auc_score, precision_score, accuracy_score, recall_score, f1_score
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
    
class FraudDetectionEnsemble:
    def __init__(self, input_dim, weights=[0.4, 0.3, 0.3]):
        self.input_dim = input_dim
        self.xgb_model = xgb.XGBClassifier(
            scale_pos_weight=10,
            max_depth=5,
            learning_rate=0.1,
            n_estimators=100,
            eval_metric='auc',  
            use_label_encoder=False,
            # early_stopping_rounds=10,
        )
        
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            max_depth=10,
            random_state=42
        )
        
        self.nn_model = FraudDetectionNN(input_dim)
        self.weights = weights
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        print(f"Training ensemble with input dimension: {self.input_dim}")
        mlflow.set_tracking_uri("databricks")
        mlflow.set_experiment("/Users/kehinde.awomuti@pwc.com/fraud_detection_train")

        # Start MLflow run
        with mlflow.start_run(run_name="fraud_detection_ensemble"):
            # Train XGBoost
            if X_val is not None:
                eval_set = [(X_train, y_train), (X_val, y_val)]
                self.xgb_model.fit(
                    X_train, y_train,
                    eval_set=eval_set,
                    verbose=False
                )
            else:
                self.xgb_model.fit(X_train, y_train)
            
            mlflow.sklearn.log_model(self.xgb_model, "xgboost_model")
            
            # Train Random Forest
            self.rf_model.fit(X_train, y_train)
            mlflow.sklearn.log_model(self.rf_model, "random_forest_model")
            
            # Train Neural Network
            X_train_tensor = torch.FloatTensor(X_train)
            y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
            
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
            X_tensor = torch.FloatTensor(X)
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

def train_fraud_detection_system(raw_data, test_size=0.2):
    print("Starting fraud detection system training...")
    
    # Initialize preprocessor
    preprocessor = TransactionPreprocessor()
    
    # Preprocess all data
    print("Preprocessing data...")
    feature_groups, labels = preprocessor.transform(raw_data, training=True)
    
    # Convert feature groups dictionary to numpy array
    X = np.concatenate([feature_groups[group] for group in sorted(feature_groups.keys())], axis=1)
    y = labels
    
    # Split into train/validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Print data statistics
    print(f"\nData statistics:")
    print(f"Total samples: {len(X)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Feature dimension: {X.shape[1]}")
    print(f"Class distribution: {np.bincount(y)}")
    print(f"Class balance: {np.mean(y):.3f}")
    print(f"\nFeature groups: {sorted(feature_groups.keys())}")
    
    # Initialize ensemble
    input_dim = X.shape[1]
    ensemble = FraudDetectionEnsemble(input_dim=input_dim)
    
    # Set up MLflow tracking
    # # mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment("/Users/kehinde.awomuti@pwc.com/fraud_detection_train")
    
    # Train and log metrics
    with mlflow.start_run(run_name="fraud_detection_ensemble"):
        # Log dataset info
        mlflow.log_params({
            'total_samples': len(X),
            'feature_dimension': X.shape[1],
            'class_balance': float(np.mean(y)),
            'test_size': test_size
        })

        # Train and log XGBoost
        print("\nTraining XGBoost...")
        ensemble.xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )
        
        xgb_train_probs = ensemble.xgb_model.predict_proba(X_train)[:, 1]
        xgb_val_probs = ensemble.xgb_model.predict_proba(X_val)[:, 1] 
        xgb_train_pred = (xgb_train_probs >= 0.5).astype(int)
        xgb_val_pred = (xgb_val_probs >= 0.5).astype(int)
        
        mlflow.log_metrics({
            'xgb_train_auc': roc_auc_score(y_train, xgb_train_probs),
            'xgb_val_auc': roc_auc_score(y_val, xgb_val_probs),
            'xgb_train_accuracy': accuracy_score(y_train, xgb_train_pred),
            'xgb_val_accuracy': accuracy_score(y_val, xgb_val_pred),
            'xgb_train_precision': precision_score(y_train, xgb_train_pred),
            'xgb_val_precision': precision_score(y_val, xgb_val_pred),
            'xgb_train_recall': recall_score(y_train, xgb_train_pred),
            'xgb_val_recall': recall_score(y_val, xgb_val_pred),
            'xgb_train_f1': f1_score(y_train, xgb_train_pred),
            'xgb_val_f1': f1_score(y_val, xgb_val_pred)
        })
        mlflow.sklearn.log_model(ensemble.xgb_model, "xgboost_model", registered_model_name = 'xgb_model')
        
        # Train and log Random Forest
        print("Training Random Forest...")
        ensemble.rf_model.fit(X_train, y_train)
        
        rf_train_probs = ensemble.rf_model.predict_proba(X_train)[:, 1]
        rf_val_probs = ensemble.rf_model.predict_proba(X_val)[:, 1]
        rf_train_pred = (rf_train_probs >= 0.5).astype(int)
        rf_val_pred = (rf_val_probs >= 0.5).astype(int)

        mlflow.log_metrics({
            'rf_train_auc': roc_auc_score(y_train, rf_train_probs),
            'rf_val_auc': roc_auc_score(y_val, rf_val_probs),
            'rf_train_accuracy': accuracy_score(y_train, rf_train_pred),
            'rf_val_accuracy': accuracy_score(y_val, rf_val_pred),
            'rf_train_precision': precision_score(y_train, rf_train_pred),
            'rf_val_precision': precision_score(y_val, rf_val_pred),
            'rf_train_recall': recall_score(y_train, rf_train_pred),
            'rf_train_f1': f1_score(y_train, rf_train_pred),
            'rf_val_f1': f1_score(y_val, rf_val_pred)
        })
        mlflow.sklearn.log_model(ensemble.rf_model, "random_forest_model", registered_model_name = 'rf_model')
        
        # Train and log Neural Network
        print("Training Neural Network...")
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1)
        
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(ensemble.nn_model.parameters(), lr=0.001)
        
        for epoch in range(10):
            # Training step
            ensemble.nn_model.train()
            optimizer.zero_grad()
            outputs = ensemble.nn_model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            
            # Validation step
            ensemble.nn_model.eval()
            with torch.no_grad():
                val_outputs = ensemble.nn_model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)
                
                # Calculate AUC for neural network
                if epoch == 9:  # Only on last epoch
                    nn_train_probs = outputs.numpy()
                    nn_val_probs = val_outputs.numpy()
                    nn_train_pred = (nn_train_probs >= 0.5).astype(int)
                    nn_val_pred = (nn_val_probs >= 0.5).astype(int)
                    
                    mlflow.log_metrics({
                        'nn_train_auc': roc_auc_score(y_train, nn_train_probs),
                        'nn_val_auc': roc_auc_score(y_val, nn_val_probs),
                        'nn_train_accuracy': accuracy_score(y_train, nn_train_pred),
                        'nn_val_accuracy': accuracy_score(y_val, nn_val_pred),
                        'nn_train_precision': precision_score(y_train, nn_train_pred),
                        'nn_val_precision': precision_score(y_val, nn_val_pred),
                        'nn_train_recall': recall_score(y_train, nn_train_pred),
                        'nn_val_recall': recall_score(y_val, nn_val_pred),
                        'nn_train_f1': f1_score(y_train, nn_train_pred),
                        'nn_val_f1': f1_score(y_val, nn_val_pred)
                    })
            mlflow.log_metrics({
                f'nn_train_loss': loss.item(),
                f'nn_val_loss': val_loss.item()
            }, step=epoch)
        
        mlflow.pytorch.log_model(ensemble.nn_model, "pytorch_model", registered_model_name = 'pytorch_model')
        
        # Log feature dimensions
        feature_dims = {group: features.shape[1] for group, features in feature_groups.items()}
        mlflow.log_dict(feature_dims, 'feature_dimensions.json')
        
        # Log example input
        input_example = X_train[:2, :]  

        # Convert the slice to a dictionary format
        input_example = {
            f"row_{i+1}": {f"feature_{j+1}": value for j, value in enumerate(row)}
            for i, row in enumerate(input_example)
        }
        mlflow.log_dict(input_example, "input_example.json")
    
    print("Training completed successfully!")

    return preprocessor, ensemble

# Define the directory containing the files
directory = '/Workspace/Users/kehinde.awomuti@pwc.com/ccfrauddetection/data'

# List to store DataFrames
df_list = []

# Iterate over files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        file_path = os.path.join(directory, filename)
        df = pd.read_csv(file_path)
        df_list.append(df)

# Concatenate all DataFrames into a single DataFrame
combined_df = pd.concat(df_list, ignore_index=True)
print(f"dimension of fraud cases --{combined_df[combined_df['TX_FRAUD']== 1].shape}")
print(f"dimension of non fraud cases --{combined_df[combined_df['TX_FRAUD'] == 0].shape}")
# df2 = pd.concat([pos_df,neg_df], axis=0) 

import warnings
warnings.filterwarnings("ignore")
print(combined_df.shape)
preprocessor, ensemble = train_fraud_detection_system(combined_df)