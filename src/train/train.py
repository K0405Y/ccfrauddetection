import sys
import os
from src.preprocessing.prep import TransactionPreprocessor
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import  train_test_split
from sklearn.metrics import roc_auc_score, precision_score, accuracy_score, recall_score, f1_score, confusion_matrix, roc_curve
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
import seaborn as sns
import warnings
import pickle as pkl
warnings.filterwarnings("ignore")

class FraudDetectionNN(nn.Module):
    def __init__(self, input_dim):
        super(FraudDetectionNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
        # Initialize weights
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        logits = self.model(x)
        return nn.functional.softmax(logits, dim=1)
        
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
        self.weights = weights

        weight_ratio = (1/0.4)
        self.class_weights = {0: 1, 1: weight_ratio}


        self.xgb_model = xgb.XGBClassifier(
            max_depth=5,
            learning_rate=0.1,
            n_estimators=100,
            eval_metric='auc',
            use_label_encoder=False,
            objective= 'binary:logistic',
            scale_pos_weight=weight_ratio
        )
        
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight=self.class_weights,
            criterion='entropy'
        )
        
        self.nn_model = FraudDetectionNN(input_dim)
      

    def _train_xgboost(self, X_train, y_train, X_val, y_val):
        """Train XGBoost model with comprehensive metrics and feature importance"""
        print("\nTraining XGBoost model...")
        
        # Configure model
        self.xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )
        
        # Calculate predictions and probabilities
        xgb_train_probs = self.xgb_model.predict_proba(X_train)[:, 1]
        xgb_val_probs = self.xgb_model.predict_proba(X_val)[:, 1]
        # Use threshold optimization for better recall
        thresholds = np.linspace(0, 1, 100)
        best_threshold = 0.5
        best_recall = 0
        
        for threshold in thresholds:
            val_preds = (xgb_val_probs >= threshold).astype(int)
            current_recall = recall_score(y_val, val_preds)
            if current_recall > best_recall:
                best_recall = current_recall
                best_threshold = threshold
        
        # Use optimized threshold for final predictions
        xgb_train_pred = (xgb_train_probs >= best_threshold).astype(int)
        xgb_val_pred = (xgb_val_probs >= best_threshold).astype(int)
        
        metrics = {
            'xgb_threshold': best_threshold,
            'xgb_train_recall': recall_score(y_train, xgb_train_pred),
            'xgb_val_recall': recall_score(y_val, xgb_val_pred),
            'xgb_train_precision': precision_score(y_train, xgb_train_pred),
            'xgb_val_precision': precision_score(y_val, xgb_val_pred),
            'xgb_train_f1': f1_score(y_train, xgb_train_pred),
            'xgb_val_f1': f1_score(y_val, xgb_val_pred),
            'xgb_train_auc': roc_auc_score(y_train, xgb_train_probs),
            'xgb_val_auc': roc_auc_score(y_val, xgb_val_probs)
        }


        # Feature importance analysis
        importance_dict = dict(zip(
            self.feature_names,
            self.xgb_model.feature_importances_
        ))
        
        # Log top 20 important features
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:20]
        xgb_importance_dict = {f'xgb_importance_{k}': v for k, v in sorted_importance}
        metrics.update(xgb_importance_dict)
        
        # SHAP analysis
        explainer = shap.TreeExplainer(self.xgb_model)
        shap_values = explainer.shap_values(X_train)
        
        # Log SHAP plots
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_train, feature_names=self.feature_names, show=False)
        plt.tight_layout()
        mlflow.log_figure(plt.gcf(), "xgb_shap_summary.png")
        plt.close()
        
        # Log confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_val, xgb_val_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('XGBoost Confusion Matrix (Validation)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        mlflow.log_figure(plt.gcf(), "xgb_confusion_matrix.png")
        plt.close()
        
        # Log ROC curve
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_val, xgb_val_probs)
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'XGBoost ROC Curve (AUC = {metrics["xgb_val_auc"]:.3f})')
        mlflow.log_figure(plt.gcf(), "xgb_roc_curve.png")
        plt.close()
        
        return metrics, importance_dict

    def _train_random_forest(self, X_train, y_train, X_val, y_val):
        """Train Random Forest model with comprehensive metrics and feature importance"""
        print("\nTraining Random Forest model...")
        
        self.rf_model.fit(X_train, y_train)
        
        # Calculate predictions and probabilities
        rf_train_probs = self.rf_model.predict_proba(X_train)[:, 1]
        rf_val_probs = self.rf_model.predict_proba(X_val)[:, 1]
         # Threshold optimization for recall
        thresholds = np.linspace(0, 1, 100)
        best_threshold = 0.5
        best_recall = 0
        
        for threshold in thresholds:
            val_preds = (rf_val_probs >= threshold).astype(int)
            current_recall = recall_score(y_val, val_preds)
            if current_recall > best_recall:
                best_recall = current_recall
                best_threshold = threshold
        
        # Use optimized threshold
        rf_train_pred = (rf_train_probs >= best_threshold).astype(int)
        rf_val_pred = (rf_val_probs >= best_threshold).astype(int)
        
        metrics = {
            'rf_threshold': best_threshold,
            'rf_train_recall': recall_score(y_train, rf_train_pred),
            'rf_val_recall': recall_score(y_val, rf_val_pred),
            'rf_train_precision': precision_score(y_train, rf_train_pred),
            'rf_val_precision': precision_score(y_val, rf_val_pred),
            'rf_train_f1': f1_score(y_train, rf_train_pred),
            'rf_val_f1': f1_score(y_val, rf_val_pred),
            'rf_train_auc': roc_auc_score(y_train, rf_train_probs),
            'rf_val_auc': roc_auc_score(y_val, rf_val_probs)
        }


        # Feature importance analysis
        importance_dict = dict(zip(
            self.feature_names,
            self.rf_model.feature_importances_
        ))
        
        # Log top 20 important features
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:20]
        rf_importance_dict = {f'rf_importance_{k}': v for k, v in sorted_importance}
        metrics.update(rf_importance_dict)
        
        # Permutation importance
        perm_importance = permutation_importance(
            self.rf_model, X_val, y_val,
            n_repeats=10,
            random_state=42
        )
        
        # Log permutation importance plot
        plt.figure(figsize=(12, 6))
        sorted_idx = perm_importance.importances_mean.argsort()
        plt.boxplot(perm_importance.importances[sorted_idx].T,
                labels=np.array(self.feature_names)[sorted_idx])
        plt.xticks(rotation=90)
        plt.title('Random Forest Permutation Importance')
        plt.tight_layout()
        mlflow.log_figure(plt.gcf(), "rf_permutation_importance.png")
        plt.close()
        
        # Log confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_val, rf_val_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Random Forest Confusion Matrix (Validation)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        mlflow.log_figure(plt.gcf(), "rf_confusion_matrix.png")
        plt.close()
        
        # Log ROC curve
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_val, rf_val_probs)
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Random Forest ROC Curve (AUC = {metrics["rf_val_auc"]:.3f})')
        mlflow.log_figure(plt.gcf(), "rf_roc_curve.png")
        plt.close()
        
        return metrics, importance_dict

    def _train_neural_network(self, X_train, y_train, X_val, y_val, epochs=20):
        """Train Neural Network with comprehensive metrics and feature importance"""
        print("\nTraining Neural Network model...")
        
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.LongTensor(y_val)
        
        # Use class weights
        weight_ratio = self.class_weights[1]
        class_weights = torch.FloatTensor([1.0, weight_ratio])

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(
            self.nn_model.parameters(),
            lr=0.001,
            weight_decay=0.01
        )
        
    # Initialize tracking
        best_recall = 0
        best_threshold = 0.5
        best_model_state = None
        history = {
            'train_loss': [], 'val_loss': [],
            'val_recall': [], 'val_precision': []
        }
        
        for epoch in range(epochs):
            # Training
            self.nn_model.train()
            optimizer.zero_grad()
            outputs = self.nn_model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            
            # Validation
            self.nn_model.eval()
            with torch.no_grad():
                val_outputs = self.nn_model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)
                val_probs = val_outputs[:, 1].numpy()
                
                # Find best threshold for recall
                for threshold in np.linspace(0, 1, 100):
                    val_preds = (val_probs >= threshold).astype(int)
                    current_recall = recall_score(y_val, val_preds)
                    if current_recall > best_recall:
                        best_recall = current_recall
                        best_threshold = threshold
                        best_model_state = self.nn_model.state_dict().copy()
                
                # Track metrics
                val_preds = (val_probs >= best_threshold).astype(int)
                history['train_loss'].append(loss.item())
                history['val_loss'].append(val_loss.item())
                history['val_recall'].append(recall_score(y_val, val_preds))
                history['val_precision'].append(precision_score(y_val, val_preds))
                
                # Log to MLflow
                mlflow.log_metrics({
                    'nn_train_loss': loss.item(),
                    'nn_val_loss': val_loss.item(),
                    'nn_val_recall': history['val_recall'][-1],
                    'nn_val_precision': history['val_precision'][-1]
                }, step=epoch)
        
        # Restore best model
        if best_model_state:
            self.nn_model.load_state_dict(best_model_state)
        
        # Final evaluation
        self.nn_model.eval()
        with torch.no_grad():
            train_outputs = self.nn_model(X_train_tensor)
            val_outputs = self.nn_model(X_val_tensor)
            train_probs = train_outputs[:, 1].numpy()
            val_probs = val_outputs[:, 1].numpy()
            train_preds = (train_probs >= best_threshold).astype(int)
            val_preds = (val_probs >= best_threshold).astype(int)
            
            metrics = {
                'nn_threshold': best_threshold,
                'nn_train_recall': recall_score(y_train, train_preds),
                'nn_val_recall': recall_score(y_val, val_preds),
                'nn_train_precision': precision_score(y_train, train_preds),
                'nn_val_precision': precision_score(y_val, val_preds),
                'nn_train_f1': f1_score(y_train, train_preds),
                'nn_val_f1': f1_score(y_val, val_preds),
                'nn_train_auc': roc_auc_score(y_train, train_probs),
                'nn_val_auc': roc_auc_score(y_val, val_probs)
            }
            
            # Training history plot
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.plot(history['train_loss'], label='Train Loss')
            plt.plot(history['val_loss'], label='Val Loss')
            plt.title('Loss History')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(history['val_recall'], label='Recall')
            plt.plot(history['val_precision'], label='Precision')
            plt.title('Validation Metrics')
            plt.legend()
            plt.tight_layout()
            mlflow.log_figure(plt.gcf(), "nn_training_history.png")
            plt.close()
            
            # Confusion matrix
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_val, val_preds)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Neural Network Confusion Matrix\nRecall: {metrics["nn_val_recall"]:.2f}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            mlflow.log_figure(plt.gcf(), "nn_confusion_matrix.png")
            plt.close()
            
            # Plot recall vs threshold curve
            plt.figure(figsize=(8, 6))
            thresholds = np.linspace(0, 1, 100)
            recalls = [recall_score(y_val, (val_probs >= t).astype(int)) for t in thresholds]
            plt.plot(thresholds, recalls)
            plt.axvline(x=best_threshold, color='r', linestyle='--', 
                    label=f'Best Threshold: {best_threshold:.2f}')
            plt.xlabel('Threshold')
            plt.ylabel('Recall')
            plt.title('Recall vs Threshold')
            plt.legend()
            plt.tight_layout()
            mlflow.log_figure(plt.gcf(), "nn_recall_threshold.png")
            plt.close()

            # Log ROC curve
            plt.figure(figsize=(8, 6))
            fpr, tpr, _ = roc_curve(y_val, val_probs)
            plt.plot(fpr, tpr)
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'Neural Network ROC Curve (AUC = {metrics["nn_val_auc"]:.2f})')
            mlflow.log_figure(plt.gcf(), "nn_roc_curve.png")
            plt.close()
                
            mlflow.log_metrics(metrics)
        

        # Feature importance analysis
        ig = IntegratedGradients(self.nn_model)
        attributions = ig.attribute(X_train_tensor[:100], target=1)
        importance_dict = dict(zip(
            self.feature_names,
            np.mean(abs(attributions.detach().numpy()), axis=0)
        ))
        # Add feature importance to metrics
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:20]
        for k, v in sorted_importance:
            metrics[f'nn_importance_{k}'] = float(v)

        # Log feature attributions
        plt.figure(figsize=(12, 6))
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:20]
        plt.bar([x[0] for x in sorted_importance], [x[1] for x in sorted_importance])
        plt.xticks(rotation=45, ha='right')
        plt.title('Neural Network Feature Importance (Integrated Gradients)')
        plt.tight_layout()
        mlflow.log_figure(plt.gcf(), "nn_feature_importance.png")
        plt.close()
        
        return metrics, importance_dict

    def train(self, X_train, y_train, X_val, y_val):
        """Train all models in the ensemble with comprehensive logging"""
        print(f"Training ensemble with input dimension: {self.input_dim}")
        
        # Train individual models
        xgb_metrics, xgb_importance = self._train_xgboost(X_train, y_train, X_val, y_val)
        rf_metrics, rf_importance = self._train_random_forest(X_train, y_train, X_val, y_val)
        nn_metrics, nn_importance = self._train_neural_network(X_train, y_train, X_val, y_val)
        
        # Get ensemble predictions
        xgb_val_probs = self.xgb_model.predict_proba(X_val)[:, 1]
        rf_val_probs = self.rf_model.predict_proba(X_val)[:, 1]
        nn_val_probs = self.nn_model(torch.FloatTensor(X_val)).detach().numpy()[:, 1]
        
        # Calculate weighted ensemble predictions
        ensemble_val_probs = (
            self.weights[0] * xgb_val_probs +
            self.weights[1] * rf_val_probs +
            self.weights[2] * nn_val_probs
        )
        ensemble_val_preds = (ensemble_val_probs >= 0.5).astype(int)
        
        # Calculate ensemble metrics
        ensemble_metrics = {
            'ensemble_val_auc': roc_auc_score(y_val, ensemble_val_probs),
            'ensemble_val_accuracy': accuracy_score(y_val, ensemble_val_preds),
            'ensemble_val_precision': precision_score(y_val, ensemble_val_preds),
            'ensemble_val_recall': recall_score(y_val, ensemble_val_preds),
            'ensemble_val_f1': f1_score(y_val, ensemble_val_preds)
        }
        
        # Calculate and log ensemble feature importance
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
        )[:20]
        
        ensemble_importance_metrics = {
            f'ensemble_importance_{k}': v
            for k, v in sorted_ensemble_importance
        }
        mlflow.log_metrics(ensemble_importance_metrics)
        
        # Log ensemble importance plot
        plt.figure(figsize=(12, 6))
        plt.bar([x[0] for x in sorted_ensemble_importance],
                [x[1] for x in sorted_ensemble_importance])
        plt.xticks(rotation=45, ha='right')
        plt.title('Ensemble Feature Importance')
        plt.tight_layout()
        mlflow.log_figure(plt.gcf(), "ensemble_feature_importance.png")
        plt.close()
        
        # Log ensemble confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_val, ensemble_val_preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Ensemble Confusion Matrix (Validation)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        mlflow.log_figure(plt.gcf(), "ensemble_confusion_matrix.png")
        plt.close()
        
        # Log ensemble ROC curve
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_val, ensemble_val_probs)
        plt.plot(fpr, tpr, label=f'Ensemble (AUC = {ensemble_metrics["ensemble_val_auc"]:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Ensemble ROC Curve')
        plt.legend()
        plt.tight_layout()
        mlflow.log_figure(plt.gcf(), "ensemble_roc_curve.png")
        plt.close()
        
        # Log comparison plot of all models
        plt.figure(figsize=(10, 6))
        models = ['XGBoost', 'Random Forest', 'Neural Network', 'Ensemble']
        metrics = ['AUC', 'F1', 'Precision', 'Recall']
        values = np.array([
            [xgb_metrics['xgb_val_auc'], xgb_metrics['xgb_val_f1'],
            xgb_metrics['xgb_val_precision'], xgb_metrics['xgb_val_recall']],
            [rf_metrics['rf_val_auc'], rf_metrics['rf_val_f1'],
            rf_metrics['rf_val_precision'], rf_metrics['rf_val_recall']],
            [nn_metrics['nn_val_auc'], nn_metrics['nn_val_f1'],
            nn_metrics['nn_val_precision'], nn_metrics['nn_val_recall']],
            [ensemble_metrics['ensemble_val_auc'], ensemble_metrics['ensemble_val_f1'],
            ensemble_metrics['ensemble_val_precision'], ensemble_metrics['ensemble_val_recall']]
        ])
        
        x = np.arange(len(metrics))
        width = 0.15
        
        for i in range(len(models)):
            plt.bar(x + i*width, values[i], width, label=models[i])
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Model Performance Comparison')
        plt.xticks(x + width*1.5, metrics)
        plt.legend(loc='lower right')
        plt.tight_layout()
        mlflow.log_figure(plt.gcf(), "model_comparison.png")
        plt.close()
        
        # Compile all metrics
        all_metrics = {
            'xgboost': xgb_metrics,
            'random_forest': rf_metrics,
            'neural_network': nn_metrics,
            'ensemble': ensemble_metrics
        }
        
        # Log all metrics
        mlflow.log_dict(all_metrics, 'all_metrics.json')
        
        # Log feature importance comparison
        feature_importance_comparison = pd.DataFrame({
            'XGBoost': pd.Series(xgb_importance),
            'Random Forest': pd.Series(rf_importance),
            'Neural Network': pd.Series(nn_importance),
            'Ensemble': pd.Series(ensemble_importance)
        })

        mlflow.log_dict(feature_importance_comparison.to_dict(), 'feature_importance_comparison.json')
        
        return all_metrics

def train_fraud_detection_system(raw_data, test_size=0.25):
    """Main training function with comprehensive logging and validation"""
    print("Starting fraud detection system training...")
    
    # Initialize preprocessor
    preprocessor = TransactionPreprocessor()
    print("Preprocessing data and engineering features...")
    
    # Get feature names
    feature_names = []
    for group in sorted(preprocessor.feature_groups.keys()):
        feature_names.extend(preprocessor.feature_groups[group])
    
    # Split data first to avoid data leakage
    train_data, val_data = train_test_split(
        raw_data, 
        test_size=test_size, 
        random_state=42,
        stratify=raw_data['TX_FRAUD']  # Ensure balanced split
    )
    
    # Process training data
    print("Processing training data...")
    feature_groups_train, y_train_resampled = preprocessor.transform(train_data, training=True)
    
    # Process validation data (without SMOTE)
    print("Processing validation data...")
    feature_groups_val = preprocessor.transform(val_data, training=False)
    y_val = val_data['TX_FRAUD'].values
    
    # Convert feature groups to numpy arrays
    X_train = np.concatenate([feature_groups_train[group] for group in preprocessor.feature_groups.keys()], axis=1)
    X_val = np.concatenate([feature_groups_val[group] for group in preprocessor.feature_groups.keys()], axis=1)
    
    # Print and log data statistics
    print("\nData Statistics:")
    data_stats = {
        'total_samples': len(raw_data),
        'training_samples': len(X_train),
        'validation_samples': len(X_val),
        'feature_dimension': X_train.shape[1],
        'original_fraud_ratio': float(raw_data['TX_FRAUD'].mean()),
        'training_fraud_ratio': float(y_train_resampled.mean()),
        'validation_fraud_ratio': float(y_val.mean())
    }
    
    for stat, value in data_stats.items():
        print(f"{stat}: {value}")
        
    # Initialize and train ensemble
    print("\nInitializing ensemble model...")
    ensemble = FraudDetectionEnsemble(
        input_dim=X_train.shape[1],
        feature_names=feature_names,
        weights=[0.4, 0.3, 0.3]  # Adjustable weights for ensemble
    )
    
    
    with mlflow.start_run(run_name="fraud_detection_ensemble", experiment_id=
                          mlflow.get_experiment_by_name("/Users/kehinde.awomuti@pwc.com/fraud_detection_train").experiment_id) as run:
        # Log dataset info
        mlflow.log_params(data_stats)

        # Log feature groups information
        feature_dims = {group: features.shape[1] for group, features in feature_groups_train.items()}
        mlflow.log_dict(feature_dims, 'feature_dimensions.json')
        
        # Train ensemble and get metrics
        print("\nTraining ensemble...")
        metrics = ensemble.train(X_train, y_train_resampled, X_val, y_val)
        
        # Log final metrics summary
        mlflow.log_dict(metrics, 'final_metrics.json')
        
        input_example = pd.DataFrame(X_train[:1], columns=feature_names)

        signature = mlflow.models.signature.infer_signature(X_train, ensemble.xgb_model.predict_proba(X_train))

        # Log XGBoost model with signature and register
        mlflow.sklearn.log_model(
            ensemble.xgb_model, 
            "xgboost_model",
            signature=signature,
            input_example=input_example,
            registered_model_name="fd_xgboost"
        )

        # Log Random Forest model with signature and register
        mlflow.sklearn.log_model(
            ensemble.rf_model, 
            "random_forest_model",
            signature=signature,
            input_example=input_example,
            registered_model_name="fd_random_forest"
        )

        # Log Neural Network model with custom wrapper, signature and register
        wrapped_nn = FraudDetectionNNWrapper(ensemble.nn_model)
        nn_signature = mlflow.models.signature.infer_signature(X_train, wrapped_nn.predict(None, pd.DataFrame(X_train, columns=feature_names)))
        mlflow.pyfunc.log_model(
            "neural_network_model",
            python_model=wrapped_nn,
            signature=nn_signature,
            input_example=input_example,
            registered_model_name="fd_torch")     

        # Create and log a summary report
        summary_report = f"""
        Fraud Detection Training Summary
        ==============================
        
        Dataset Statistics:
        - Total samples: {data_stats['total_samples']:,}
        - Training samples: {data_stats['training_samples']:,}
        - Validation samples: {data_stats['validation_samples']:,}
        - Feature dimension: {data_stats['feature_dimension']}
        - Original fraud ratio: {data_stats['original_fraud_ratio']:.3%}
        - Training fraud ratio: {data_stats['training_fraud_ratio']:.3%}
        - Validation fraud ratio: {data_stats['validation_fraud_ratio']:.3%}
        
        Model Performance (Validation):
        - XGBoost AUC: {metrics['xgboost']['xgb_val_auc']:.3f}
        - Random Forest AUC: {metrics['random_forest']['rf_val_auc']:.3f}
        - Neural Network AUC: {metrics['neural_network']['nn_val_auc']:.3f}
        - Ensemble AUC: {metrics['ensemble']['ensemble_val_auc']:.3f}
        
        Feature Groups:
        {'-' * 40}
        """ + '\n'.join([f"- {group}: {dims} features" for group, dims in feature_dims.items()])
        
        # Log summary report
        with open("training_summary.txt", "w") as f:
            f.write(summary_report)
        mlflow.log_artifact("training_summary.txt")
        
        print("\nTraining completed successfully!")
        print(f"MLflow run ID: {run.info.run_id}")
        
        return preprocessor, ensemble, metrics    

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
pos_df = combined_df[combined_df['TX_FRAUD'] == 1].iloc[:30]
neg_df = combined_df[combined_df['TX_FRAUD'] == 0].iloc[:5000]
df2 = pd.concat([pos_df, neg_df], ignore_index=True, axis= 0)
preprocessor, ensemble, metrics = train_fraud_detection_system(combined_df)