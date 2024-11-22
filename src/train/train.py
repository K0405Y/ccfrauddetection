import sys
import os
from src.preprocessing.prep import TransactionPreprocessor
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import  train_test_split
from sklearn.metrics import roc_auc_score, precision_score, accuracy_score, recall_score, f1_score
from sklearn.inspection import permutation_importance
import xgboost as xgb
import torch
import torch.nn as nn 
import mlflow
import mlflow.pytorch
import mlflow.sklearn
import shap
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
import warnings
import pickle as pkl
warnings.filterwarnings("ignore")

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
        pos_prob = self.model(x)
        neg_prob = 1 - pos_prob
        probs = torch.cat([neg_prob, pos_prob], dim=1)
        return probs
    
class FraudDetectionNNWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
        
    def predict(self, context, model_input):
        self.model.eval()
        with torch.no_grad():
            # Convert input to tensor
            if isinstance(model_input, pd.DataFrame):
                X = torch.tensor(model_input.values, dtype=torch.float32)
            else:
                X = torch.tensor(model_input, dtype=torch.float32)
                
            # Get predictions
            outputs = self.model(X)
            # Convert to numpy array with shape (n_samples, 2)
            probabilities = outputs.numpy()
            return probabilities
            
class FraudDetectionEnsemble:
    def __init__(self, input_dim, feature_names=None, weights=[0.4, 0.3, 0.3]):
        self.input_dim = input_dim
        self.feature_names = feature_names or [f'feature_{i}' for i in range(input_dim)]
        self.xgb_model = xgb.XGBClassifier(
            max_depth=5,
            learning_rate=0.1,
            n_estimators=100,
            eval_metric='auc',
            use_label_encoder=False,
            objective= 'binary:logistic'
        )
        
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            max_depth=10,
            random_state=42,
        )
        
        self.nn_model = FraudDetectionNN(input_dim)
        self.weights = weights

    def _train_xgboost(self, X_train, y_train, X_val, y_val):
        """Train XGBoost model and log metrics and feature importance"""
        self.xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )
        
        # Performance metrics
        xgb_train_probs = self.xgb_model.predict_proba(X_train)[:, 1]
        xgb_val_probs = self.xgb_model.predict_proba(X_val)[:, 1]
        xgb_train_pred = (xgb_train_probs >= 0.5).astype(int)
        xgb_val_pred = (xgb_val_probs >= 0.5).astype(int)
        
        metrics = {
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
        }
        
        # Feature importance
        importance_dict = dict(zip(
            self.feature_names,
            self.xgb_model.feature_importances_
        ))
        
        # Log top 20 most important features
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:20]
        xgb_importance_dict = {f'xgb_importance_{k}': v for k, v in sorted_importance}
        metrics.update(xgb_importance_dict)
        
        # Log feature importance plot using SHAP
        explainer = shap.TreeExplainer(self.xgb_model)
        shap_values = explainer.shap_values(X_train)
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_train, feature_names=self.feature_names, show=False)
        plt.tight_layout()
        mlflow.log_figure(plt.gcf(), "xgb_shap_summary.png")
        plt.close()
        
        # Log model signature
        input_example = X_train[:1, :]  
        # Convert the slice to a dictionary format
        # input_example = {
        #     {i+1}: {{j+1}: value for j, value in enumerate(row)}
        #     for i, row in enumerate(input_example)
        # # }
        signature = mlflow.models.infer_signature(input_example, self.xgb_model.predict(input_example))
        mlflow.sklearn.log_model(self.xgb_model, "xgboost_model", registered_model_name='xgb_model', signature = signature)
        mlflow.log_metrics(metrics)

        return metrics, importance_dict

    def _train_random_forest(self, X_train, y_train, X_val, y_val):
        """Train Random Forest model and log metrics and feature importance"""
        self.rf_model.fit(X_train, y_train)
        
        # Performance metrics
        rf_train_probs = self.rf_model.predict_proba(X_train)[:, 1]
        rf_val_probs = self.rf_model.predict_proba(X_val)[:, 1]
        rf_train_pred = (rf_train_probs >= 0.5).astype(int)
        rf_val_pred = (rf_val_probs >= 0.5).astype(int)
        
        metrics = {
            'rf_train_auc': roc_auc_score(y_train, rf_train_probs),
            'rf_val_auc': roc_auc_score(y_val, rf_val_probs),
            'rf_train_accuracy': accuracy_score(y_train, rf_train_pred),
            'rf_val_accuracy': accuracy_score(y_val, rf_val_pred),
            'rf_train_precision': precision_score(y_train, rf_train_pred),
            'rf_val_precision': precision_score(y_val, rf_val_pred),
            'rf_train_recall': recall_score(y_train, rf_train_pred),
            'rf_val_recall': recall_score(y_val, rf_val_pred),
            'rf_train_f1': f1_score(y_train, rf_train_pred),
            'rf_val_f1': f1_score(y_val, rf_val_pred)
        }
        
        # Feature importance
        importance_dict = dict(zip(
            self.feature_names,
            self.rf_model.feature_importances_
        ))
        
        # Log top 20 most important features
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:20]
        rf_importance_dict = {f'rf_importance_{k}': v for k, v in sorted_importance}
        metrics.update(rf_importance_dict)
        
        # Create and log permutation importance plot
        perm_importance = permutation_importance(
            self.rf_model, X_val, y_val, n_repeats=10, random_state=42
        )
        
        plt.figure(figsize=(10, 6))
        sorted_idx = perm_importance.importances_mean.argsort()
        plt.boxplot(perm_importance.importances[sorted_idx].T,
                   labels=np.array(self.feature_names)[sorted_idx])
        plt.xticks(rotation=90)
        plt.tight_layout()
        mlflow.log_figure(plt.gcf(), "rf_permutation_importance.png")
        plt.close()

        # Log input example
        input_example = X_train[:1, :]  

        signature = mlflow.models.infer_signature(input_example, self.rf_model.predict_proba(input_example))
        mlflow.sklearn.log_model(self.rf_model, "random_forest_model", registered_model_name='rf_model', signature = signature)
        mlflow.log_metrics(metrics)

        return metrics, importance_dict

    def _train_neural_network(self, X_train, y_train, X_val, y_val, epochs=10):
        """Train Neural Network model and log metrics and feature importance"""
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1)
        
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.nn_model.parameters(), lr=0.001)
        
        metrics = {}
        for epoch in range(epochs):
            # Training step
            self.nn_model.train()
            optimizer.zero_grad()
            outputs = self.nn_model(X_train_tensor)
            loss = criterion(outputs[:, 1:], y_train_tensor)
            loss.backward()
            optimizer.step()
            
            # Validation step
            self.nn_model.eval()
            with torch.no_grad():
                val_outputs = self.nn_model(X_val_tensor)
                val_loss = criterion(val_outputs[:,1:], y_val_tensor)
                
                mlflow.log_metrics({
                    f'nn_train_loss': loss.item(),
                    f'nn_val_loss': val_loss.item()
                }, step=epoch)
                
                if epoch == epochs - 1:
                    nn_train_probs = outputs[:, 1].numpy()
                    nn_val_probs = val_outputs[:, 1].numpy()
                    nn_train_pred = (nn_train_probs >= 0.5).astype(int)
                    nn_val_pred = (nn_val_probs >= 0.5).astype(int)
                    
                    metrics = {
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
                    }
        
        # Feature importance using Integrated Gradients
        ig = IntegratedGradients(self.nn_model)
        attributions = ig.attribute(X_train_tensor[:100], target=1)  # Use subset for efficiency
        importance_dict = dict(zip(
            self.feature_names,
            np.mean(abs(attributions.detach().numpy()), axis=0)
        ))
        
        # Log top 20 most important features
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:20]
        nn_importance_dict = {f'nn_importance_{k}': v for k, v in sorted_importance}
        metrics.update(nn_importance_dict)
        
        # Create and log attribution plot
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(sorted_importance)), [x[1] for x in sorted_importance])
        plt.xticks(range(len(sorted_importance)), [x[0] for x in sorted_importance], rotation=90)
        plt.title('Neural Network Feature Attribution (Integrated Gradients)')
        plt.tight_layout()
        mlflow.log_figure(plt.gcf(), "nn_feature_attribution.png")
        plt.close()
        
        wrapped_model = FraudDetectionNNWrapper(self.nn_model)
        # Create an example input
        example_input = X_train[:1]
        signature = mlflow.models.infer_signature( example_input, 
                                                  np.array([[0.43989954, 0.56010046]]))
        
        # Log model with custom wrapper
        mlflow.pyfunc.log_model(
            artifact_path="pytorch_model",
            python_model=wrapped_model,
            signature = signature,
            registered_model_name="pytorch_model"
        )        

        return metrics, importance_dict

    def train(self, X_train, y_train, X_val, y_val):
        """Train all models in the ensemble"""
        print(f"Training ensemble with input dimension: {self.input_dim}")
        
        print("\nTraining XGBoost...")
        xgb_metrics, xgb_importance = self._train_xgboost(X_train, y_train, X_val, y_val)
        
        print("Training Random Forest...")
        rf_metrics, rf_importance = self._train_random_forest(X_train, y_train, X_val, y_val)
        
        print("Training Neural Network...")
        nn_metrics, nn_importance = self._train_neural_network(X_train, y_train, X_val, y_val)
        
        # Combine feature importances using ensemble weights
        ensemble_importance = {}
        for feature in self.feature_names:
            ensemble_importance[feature] = (
                self.weights[0] * xgb_importance.get(feature, 0) +
                self.weights[1] * rf_importance.get(feature, 0) +
                self.weights[2] * nn_importance.get(feature, 0)
            )
        
        # Log ensemble feature importance
        sorted_ensemble_importance = sorted(
            ensemble_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:15]
        
        ensemble_importance_metrics = {
            f'ensemble_importance_{k}': v 
            for k, v in sorted_ensemble_importance
        }
        mlflow.log_metrics(ensemble_importance_metrics)
        
        # Create and log ensemble importance plot
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(sorted_ensemble_importance)), 
                [x[1] for x in sorted_ensemble_importance])
        plt.xticks(range(len(sorted_ensemble_importance)), 
                   [x[0] for x in sorted_ensemble_importance], 
                   rotation=90)
        plt.title('Ensemble Feature Importance')
        plt.tight_layout()
        mlflow.log_figure(plt.gcf(), "ensemble_feature_importance.png")
        plt.close()
        
        return {
            'xgboost': xgb_metrics,
            'random_forest': rf_metrics,
            'neural_network': nn_metrics
        }

def train_fraud_detection_system(raw_data, test_size=0.2):
    """Main training function that handles data preprocessing and model training"""
    print("Starting fraud detection system training...")
    
    # Initialize preprocessor and prepare data
    preprocessor = TransactionPreprocessor()
    print("Preprocessing data and engineering features...")
    feature_groups, labels = preprocessor.transform(raw_data, training=True)

    #save feature groups for inference purpose
    with open('/Workspace/Users/kehinde.awomuti@pwc.com/ccfrauddetection/features/feature_groups.pkl', 'wb') as file:
        pkl.dump(feature_groups, file)

    # Convert feature groups to numpy array
    X = np.concatenate([feature_groups[group] for group in sorted(feature_groups.keys())], axis=1)
    y = labels
    
    # Split data
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

    ensemble = FraudDetectionEnsemble(input_dim=X.shape[1])
    
    # Set up MLflow tracking
    mlflow.set_experiment("/Users/kehinde.awomuti@pwc.com/fraud_detection_train")
    
    with mlflow.start_run(run_name="fraud_detection_ensemble"):
        # Log dataset info
        mlflow.log_params({
            'total_samples': len(X),
            'feature_dimension': X.shape[1],
            'class_balance': float(np.mean(y)),
            'test_size': test_size
        })
        
        # Train ensemble
        metrics = ensemble.train(X_train, y_train, X_val, y_val)    

        mlflow.log_dict(metrics, 'all_metrics.json')

        # Log feature dimensions
        feature_dims = {group: features.shape[1] for group, features in feature_groups.items()}
        mlflow.log_dict(feature_dims, 'feature_dimensions.json')        
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
print(f"dimension of all cases --{combined_df.shape}")
preprocessor, ensemble = train_fraud_detection_system(combined_df)