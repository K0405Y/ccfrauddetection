import sys
import os
from src.preprocessing.prep import TransactionPreprocessor
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import  train_test_split, TimeSeriesSplit
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
from scipy.optimize import minimize
warnings.filterwarnings("ignore")

class FraudDetectionNN(nn.Module):
    def __init__(self, input_dim):
        super(FraudDetectionNN, self).__init__()
        # Modified architecture without BatchNorm for small batches
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
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

        # Increase class weight for minority class
        weight_ratio = (1/0.026)  # Based on original class distribution
        self.class_weights = {0: 1, 1: weight_ratio}

        # Modified XGBoost with better parameters for imbalanced data
        self.xgb_model = xgb.XGBClassifier(
            max_depth=4,  # Reduced to prevent overfitting
            learning_rate=0.01,  # Slower learning rate
            n_estimators=500,  # More trees
            min_child_weight=10,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=1,
            reg_alpha=1,
            reg_lambda=2,
            # eval_metric=['auc', 'aucpr'],  # Added PR-AUC metric
            use_label_encoder=False,
            objective='binary:logistic',
            scale_pos_weight=weight_ratio,
            tree_method='hist'  # Faster training
        )
        
        # Modified Random Forest with better parameters for imbalanced data
        self.rf_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=4,
            min_samples_split=10,
            random_state=42,
            class_weight='balanced_subsample',
            bootstrap=True,
            oob_score=True,
            n_jobs=-1  # Use all cores
        )
        
        self.nn_model = FraudDetectionNN(input_dim)
        
        # Initialize thresholds with focus on recall
        self.thresholds = {
            'xgboost': 0.3,
            'random_forest': 0.3,
            'neural_network': 0.3,
            'ensemble': 0.3
        }

    def _optimize_threshold(self, y_true, y_prob, metric='f1', threshold_range=None):
        """Enhanced threshold optimization with focus on fraud detection"""
        if threshold_range is None:
            threshold_range = np.linspace(0.1, 0.9, 100)
        
        best_threshold = 0.3  # Default threshold favoring recall
        best_score = 0
        
        for threshold in threshold_range:
            y_pred = (y_prob >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_true, y_pred)
            elif metric == 'precision':
                score = precision_score(y_true, y_pred)
            elif metric == 'recall':
                score = recall_score(y_true, y_pred)
            elif metric == 'balanced':
                prec = precision_score(y_true, y_pred)
                rec = recall_score(y_true, y_pred)
                score = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
            else:
                raise ValueError(f"Unsupported metric: {metric}")
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        return best_threshold, best_score

    def _train_xgboost(self, X_train, y_train, X_val, y_val):
        """Enhanced XGBoost training with early stopping and custom eval metric"""
        print("\nTraining XGBoost model...")
        
        self.xgb_model.set_params(eval_metric='aucpr')

        # Train with early stopping
        self.xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            early_stopping_rounds=30,
            verbose=100
        )
        
        # Get validation predictions with optimal threshold
        val_probs = self.xgb_model.predict_proba(X_val)[:, 1]
        optimal_threshold, _ = self._optimize_threshold(y_val, val_probs, metric='f1')
        val_preds = (val_probs >= optimal_threshold).astype(int)
        
        metrics = {
            'xgboost_val_auc': roc_auc_score(y_val, val_probs),
            'xgboost_val_accuracy': accuracy_score(y_val, val_preds),
            'xgboost_val_precision': precision_score(y_val, val_preds),
            'xgboost_val_recall': recall_score(y_val, val_preds),
            'xgboost_val_f1': f1_score(y_val, val_preds),
            'xgboost_threshold': optimal_threshold
        }
        
        return metrics, self.xgb_model.feature_importances_

    def _train_random_forest(self, X_train, y_train, X_val, y_val):
        """Enhanced Random Forest training with optimal threshold search"""
        print("\nTraining Random Forest model...")
        
        # Train the model
        self.rf_model.fit(X_train, y_train)
        
        # Get validation predictions with optimal threshold
        val_probs = self.rf_model.predict_proba(X_val)[:, 1]
        optimal_threshold, _ = self._optimize_threshold(y_val, val_probs, metric='recall')
        val_preds = (val_probs >= optimal_threshold).astype(int)
        
        metrics = {
            'random_forest_val_auc': roc_auc_score(y_val, val_probs),
            'random_forest_val_accuracy': accuracy_score(y_val, val_preds),
            'random_forest_val_precision': precision_score(y_val, val_preds),
            'random_forest_val_recall': recall_score(y_val, val_preds),
            'random_forest_val_f1': f1_score(y_val, val_preds),
            'random_forest_oob_score': self.rf_model.oob_score_,
            'random_forest_threshold': optimal_threshold
        }
        
        return metrics, self.rf_model.feature_importances_

    def _train_neural_network(self, X_train, y_train, X_val, y_val):
        """Enhanced Neural Network training with improved batch handling and monitoring"""
        print("\nTraining Neural Network model...")
        
        # Convert data to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        
        # Define training parameters with adjusted batch size
        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1, self.class_weights[1]]))
        optimizer = torch.optim.Adam(self.nn_model.parameters(), lr=0.001, weight_decay=1e-5)
        
        # Adjust batch size based on dataset size
        batch_size = min(32, len(X_train) // 10)  # Ensure at least 10 batches
        batch_size = max(batch_size, 4)  # Ensure batch size is at least 4
        
        n_epochs = 100
        best_val_auc = 0
        patience = 10
        patience_counter = 0
        
        # Create data loaders with adjusted batch size
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True  # Drop last batch if incomplete
        )
        
        # Training loop with improved monitoring
        for epoch in range(n_epochs):
            self.nn_model.train()
            total_loss = 0
            batch_count = 0
            
            for X_batch, y_batch in train_loader:
                if len(X_batch) < 2:  # Skip batches that are too small
                    continue
                    
                optimizer.zero_grad()
                output = self.nn_model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.nn_model.parameters(), max_norm=1.0)
                
                optimizer.step()
                total_loss += loss.item()
                batch_count += 1
            
            avg_loss = total_loss / batch_count if batch_count > 0 else float('inf')
            
            # Validation
            self.nn_model.eval()
            with torch.no_grad():
                val_output = self.nn_model(X_val_tensor)
                val_probs = val_output.numpy()[:, 1]
                optimal_threshold, _ = self._optimize_threshold(y_val, val_probs, metric='recall')
                val_preds = (val_probs >= optimal_threshold).astype(int)
                val_auc = roc_auc_score(y_val, val_probs)
                val_f1 = f1_score(y_val, val_preds)
            
            # Early stopping with improved criteria
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                best_state = self.nn_model.state_dict()
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}, "
                      f"Val AUC: {val_auc:.4f}, Val F1: {val_f1:.4f}")
        
        # Restore best model
        self.nn_model.load_state_dict(best_state)
        
        # Final validation metrics
        self.nn_model.eval()
        with torch.no_grad():
            val_output = self.nn_model(X_val_tensor)
            val_probs = val_output.numpy()[:, 1]
            val_preds = (val_probs >= optimal_threshold).astype(int)
        
        metrics = {
            'neural_network_val_auc': roc_auc_score(y_val, val_probs),
            'neural_network_val_accuracy': accuracy_score(y_val, val_preds),
            'neural_network_val_precision': precision_score(y_val, val_preds),
            'neural_network_val_recall': recall_score(y_val, val_preds),
            'neural_network_val_f1': f1_score(y_val, val_preds),
            'neural_network_threshold': optimal_threshold
        }
        
        # Feature importance using integrated gradients
        ig = IntegratedGradients(self.nn_model)
        attributions = ig.attribute(X_val_tensor, target=1)
        importance = attributions.mean(dim=0).abs().numpy()
        
        return metrics, importance

    def _optimize_weights(self, X_val, y_val):
        """Enhanced ensemble weight optimization with focus on fraud detection"""
        print("\nOptimizing ensemble weights...")
        
        # Get predictions from each model
        xgb_probs = self.xgb_model.predict_proba(X_val)[:, 1]
        rf_probs = self.rf_model.predict_proba(X_val)[:, 1]
        nn_probs = self.nn_model(torch.FloatTensor(X_val)).detach().numpy()[:, 1]
        
        def objective(weights):
            # Normalize weights
            weights = np.array(weights)
            weights = weights / weights.sum()
            
            # Calculate weighted probabilities
            ensemble_probs = (
                weights[0] * xgb_probs +
                weights[1] * rf_probs +
                weights[2] * nn_probs
            )
            
            # Use optimal threshold
            optimal_threshold, _ = self._optimize_threshold(y_val, ensemble_probs, metric='f1')
            ensemble_preds = (ensemble_probs >= optimal_threshold).astype(int)
            
            # Optimize for both AUC and recall
            auc_score = roc_auc_score(y_val, ensemble_probs)
            recall = recall_score(y_val, ensemble_preds)
            f1 = f1_score(y_val, ensemble_preds)
            
            # Combined score with emphasis on recall
            return -(0.4 * auc_score + 0.3 * recall + 0.3 * f1)
        
        # Optimize weights
        initial_weights = np.array(self.weights)
        bounds = [(0, 1)] * 3
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        
        result = minimize(
            objective,
            initial_weights,
            bounds=bounds,
            constraints=constraints,
            method='SLSQP'
        )
        
        # Update weights
        self.weights = list(result.x / result.x.sum())
        print(f"Optimized weights: XGBoost={self.weights[0]:.3f}, "
              f"RF={self.weights[1]:.3f}, NN={self.weights[2]:.3f}")
        
        return self.weights

    def _optimize_model_thresholds(self, X_val, y_val, metric='f1'):
        """Enhanced threshold optimization for all models"""
        print("\nOptimizing classification thresholds...")
        
        # Get predictions from each model
        xgb_probs = self.xgb_model.predict_proba(X_val)[:, 1]
        rf_probs = self.rf_model.predict_proba(X_val)[:, 1]
        nn_probs = self.nn_model(torch.FloatTensor(X_val)).detach().numpy()[:, 1]
        
        # Optimize thresholds for individual models
        self.thresholds['xgboost'], xgb_score = self._optimize_threshold(y_val, xgb_probs, metric)
        self.thresholds['random_forest'], rf_score = self._optimize_threshold(y_val, rf_probs, metric)
        self.thresholds['neural_network'], nn_score = self._optimize_threshold(y_val, nn_probs, metric)
        
        # Get ensemble predictions with current weights
        ensemble_probs = (
            self.weights[0] * xgb_probs +
            self.weights[1] * rf_probs +
            self.weights[2] * nn_probs
        )
        
        # Optimize ensemble threshold
        self.thresholds['ensemble'], ensemble_score = self._optimize_threshold(y_val, ensemble_probs, metric)
        
        print(f"\nOptimized Thresholds ({metric}):")
        print(f"XGBoost: {self.thresholds['xgboost']:.3f} (score: {xgb_score:.3f})")
        print(f"Random Forest: {self.thresholds['random_forest']:.3f} (score: {rf_score:.3f})")
        print(f"Neural Network: {self.thresholds['neural_network']:.3f} (score: {nn_score:.3f})")
        print(f"Ensemble: {self.thresholds['ensemble']:.3f} (score: {ensemble_score:.3f})")
        
        return self.thresholds

    def train(self, X_train, y_train, X_val, y_val):
        """Enhanced training with improved monitoring and validation"""
        print(f"Training ensemble with input dimension: {self.input_dim}")
        
        # Train individual models with enhanced monitoring
        xgb_metrics, xgb_importance = self._train_xgboost(X_train, y_train, X_val, y_val)
        rf_metrics, rf_importance = self._train_random_forest(X_train, y_train, X_val, y_val)
        nn_metrics, nn_importance = self._train_neural_network(X_train, y_train, X_val, y_val)
        
        # Optimize model weights based on validation performance
        print("\nOptimizing ensemble weights...")
        self._optimize_weights(X_val, y_val)
        
        # Optimize thresholds with focus on fraud detection
        print("\nOptimizing classification thresholds...")
        self._optimize_model_thresholds(X_val, y_val, metric='f1')
        
        # Get ensemble predictions using optimized weights and thresholds
        ensemble_val_probs = self.predict_proba(X_val)[:, 1]
        ensemble_val_preds = (ensemble_val_probs >= self.thresholds['ensemble']).astype(int)
        
        # Calculate comprehensive ensemble metrics
        ensemble_metrics = {
            'ensemble_val_auc': roc_auc_score(y_val, ensemble_val_probs),
            'ensemble_val_accuracy': accuracy_score(y_val, ensemble_val_preds),
            'ensemble_val_precision': precision_score(y_val, ensemble_val_preds),
            'ensemble_val_recall': recall_score(y_val, ensemble_val_preds),
            'ensemble_val_f1': f1_score(y_val, ensemble_val_preds),
            'ensemble_threshold': self.thresholds['ensemble']
        }
        
        # Calculate feature importance rankings
        feature_importance = {
            'xgboost': {name: score for name, score in zip(self.feature_names, xgb_importance)},
            'random_forest': {name: score for name, score in zip(self.feature_names, rf_importance)},
            'neural_network': {name: score for name, score in zip(self.feature_names, nn_importance)}
        }
        
        # Compile all metrics and information
        all_metrics = {
            'xgboost': xgb_metrics,
            'random_forest': rf_metrics,
            'neural_network': nn_metrics,
            'ensemble': ensemble_metrics,
            'weights': self.weights,
            'thresholds': self.thresholds,
            'feature_importance': feature_importance
        }
        
        return all_metrics

    def predict_proba(self, X):
        """Get probability predictions using optimized weights"""
        X_tensor = torch.FloatTensor(X)
        
        # Get predictions from each model
        xgb_probs = self.xgb_model.predict_proba(X)[:, 1]
        rf_probs = self.rf_model.predict_proba(X)[:, 1]
        nn_probs = self.nn_model(X_tensor).detach().numpy()[:, 1]
        
        # Calculate weighted probabilities using optimized weights
        weighted_probs = (
            self.weights[0] * xgb_probs +
            self.weights[1] * rf_probs +
            self.weights[2] * nn_probs
        )
        
        return np.vstack((1 - weighted_probs, weighted_probs)).T

    def predict(self, X, threshold=None):
        """Get class predictions using optimized weights and thresholds"""
        if threshold is None:
            threshold = self.thresholds['ensemble']
            
        probas = self.predict_proba(X)
        return (probas[:, 1] >= threshold).astype(int)
    
    def predict_model(self, X, model_name):
        """Get predictions from a specific model using its optimized threshold"""
        if model_name not in self.thresholds:
            raise ValueError(f"Unknown model: {model_name}")
        
        if model_name == 'xgboost':
            probs = self.xgb_model.predict_proba(X)[:, 1]
            threshold = self.thresholds['xgboost']
        elif model_name == 'random_forest':
            probs = self.rf_model.predict_proba(X)[:, 1]
            threshold = self.thresholds['random_forest']
        elif model_name == 'neural_network':
            probs = self.nn_model(torch.FloatTensor(X)).detach().numpy()[:, 1]
            threshold = self.thresholds['neural_network']
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        return (probs >= threshold).astype(int)

    def get_feature_importance(self, method='ensemble'):
        """Get feature importance scores using specified method"""
        if method == 'ensemble':
            # Combine feature importance from all models
            xgb_importance = self.xgb_model.feature_importances_
            rf_importance = self.rf_model.feature_importances_
            
            # Get neural network importance using integrated gradients
            self.nn_model.eval()
            with torch.no_grad():
                ig = IntegratedGradients(self.nn_model)
                attributions = ig.attribute(torch.zeros((1, self.input_dim)), target=1)
                nn_importance = attributions.mean(dim=0).abs().numpy()
            
            # Normalize importances
            xgb_importance = xgb_importance / xgb_importance.sum()
            rf_importance = rf_importance / rf_importance.sum()
            nn_importance = nn_importance / nn_importance.sum()
            
            # Weighted average of importance scores
            ensemble_importance = (
                self.weights[0] * xgb_importance +
                self.weights[1] * rf_importance +
                self.weights[2] * nn_importance
            )
            
            return {
                name: score for name, score in zip(self.feature_names, ensemble_importance)
            }
        elif method in ['xgboost', 'random_forest', 'neural_network']:
            return getattr(self, f'{method}_feature_importance')()
        else:
            raise ValueError(f"Unsupported importance method: {method}")

    def get_optimized_parameters(self):
        """Return current optimized parameters"""
        return {
            'weights': self.weights,
            'thresholds': self.thresholds,
            'model_parameters': {
                'xgboost': self.xgb_model.get_params(),
                'random_forest': self.rf_model.get_params()
            }
        }

    def save_model(self, path):
        """Save the ensemble model to disk"""
        model_data = {
            'weights': self.weights,
            'thresholds': self.thresholds,
            'feature_names': self.feature_names,
            'xgb_model': self.xgb_model,
            'rf_model': self.rf_model,
            'nn_model_state': self.nn_model.state_dict(),
            'input_dim': self.input_dim
        }
        torch.save(model_data, path)

    @classmethod
    def load_model(cls, path):
        """Load the ensemble model from disk"""
        model_data = torch.load(path)
        
        # Initialize ensemble with saved parameters
        ensemble = cls(
            input_dim=model_data['input_dim'],
            feature_names=model_data['feature_names'],
            weights=model_data['weights']
        )
        
        # Load individual models
        ensemble.xgb_model = model_data['xgb_model']
        ensemble.rf_model = model_data['rf_model']
        ensemble.nn_model.load_state_dict(model_data['nn_model_state'])
        ensemble.thresholds = model_data['thresholds']
        
        return ensemble


def train_fraud_detection_system(raw_data, test_size=0.25):
    """Main training function with improved validation and ensemble strategy"""
    print("Starting fraud detection system training...")
    
    # Initialize preprocessor
    preprocessor = TransactionPreprocessor()
    
    # Convert TX_DATETIME to datetime format before validation
    try:
        raw_data['TX_DATETIME'] = pd.to_datetime(raw_data['TX_DATETIME'])
    except Exception as e:
        print(f"Error converting TX_DATETIME to datetime format: {str(e)}")
        raw_data['TX_DATETIME'] = pd.to_datetime(raw_data['TX_DATETIME'], format='%Y-%m-%d %H:%M:%S')
    
    # Validate input data first
    print("Validating input data...")
    issues = preprocessor.validate_features(raw_data)
    if issues:
        print("Data validation issues found:")
        for issue in issues:
            print(f"- {issue}")
        raise ValueError("Data validation failed. Please fix the issues before proceeding.")
    
    print("Preprocessing data and engineering features...")

    # Get feature names and metadata
    feature_names = preprocessor.get_feature_names()
    feature_metadata = preprocessor.get_feature_metadata()
    
    # Sort by time and create time-based split
    raw_data['TX_DATETIME'] = pd.to_datetime(raw_data['TX_DATETIME'])
    raw_data = raw_data.sort_values('TX_DATETIME')
    
    # Use time-based split with validation buffer
    train_end_date = raw_data['TX_DATETIME'].quantile(0.6)
    val_start_date = raw_data['TX_DATETIME'].quantile(0.6)
    val_end_date = raw_data['TX_DATETIME'].quantile(0.8)
    test_start_date = raw_data['TX_DATETIME'].quantile(0.8)
    
    train_data = raw_data[raw_data['TX_DATETIME'] <= train_end_date]
    val_data = raw_data[(raw_data['TX_DATETIME'] > val_start_date) & 
                        (raw_data['TX_DATETIME'] <= val_end_date)]
    test_data = raw_data[raw_data['TX_DATETIME'] > test_start_date]
    
    # Process data for each split
    print("\nProcessing training data...")
    feature_groups_train, y_train_resampled = preprocessor.transform(train_data, training=True)
    
    print("Processing validation data...")
    feature_groups_val = preprocessor.transform(val_data, training=False)
    y_val = val_data['TX_FRAUD'].values
    
    print("Processing test data...")
    feature_groups_test = preprocessor.transform(test_data, training=False)
    y_test = test_data['TX_FRAUD'].values
    
    # Convert feature groups to numpy arrays
    X_train = np.concatenate([feature_groups_train[group] for group in preprocessor.feature_groups.keys()], axis=1)
    X_val = np.concatenate([feature_groups_val[group] for group in preprocessor.feature_groups.keys()], axis=1)
    X_test = np.concatenate([feature_groups_test[group] for group in preprocessor.feature_groups.keys()], axis=1)
    
    # Print and log data statistics
    print("\nData Statistics:")
    data_stats = {
        'total_samples': len(raw_data),
        'training_samples': len(X_train),
        'validation_samples': len(X_val),
        'test_samples': len(X_test),
        'feature_dimension': X_train.shape[1],
        'original_fraud_ratio': float(raw_data['TX_FRAUD'].mean()),
        'training_fraud_ratio': float(y_train_resampled.mean()),
        'validation_fraud_ratio': float(y_val.mean()),
        'test_fraud_ratio': float(y_test.mean())
    }
    
    for stat, value in data_stats.items():
        print(f"{stat}: {value}")
    
    # Initialize ensemble with dynamic weight optimization
    print("\nInitializing ensemble model...")
    ensemble = FraudDetectionEnsemble(
        input_dim=X_train.shape[1],
        feature_names=feature_names
    )
    
    with mlflow.start_run(run_name="fraud_detection_ensemble", 
                          experiment_id= mlflow.get_experiment_by_name('/Users/kehinde.awomuti@pwc.com/fraud_detection_train').experiment_id) as run:
        # Log dataset info and feature metadata
        mlflow.log_params(data_stats)
        mlflow.log_dict(feature_metadata, 'feature_metadata.json')
        
        # Train ensemble with validation monitoring
        print("\nTraining ensemble...")
        metrics = ensemble.train(X_train, y_train_resampled, X_val, y_val)
        
        # Optimize ensemble weights using validation performance
        print("\nOptimizing ensemble weights...")
        ensemble._optimize_weights(X_val, y_val)
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_metrics = evaluate_ensemble(ensemble, X_test, y_test)
        metrics['test'] = test_metrics
        
        # Log final metrics summary
        mlflow.log_dict(metrics, 'final_metrics.json')
        
        # Prepare model artifacts
        input_example = pd.DataFrame(X_train[:1], columns=feature_names)
        signature = mlflow.models.signature.infer_signature(
            X_train, 
            ensemble.xgb_model.predict_proba(X_train)
        )
        
        # Log individual models
        log_models(ensemble, signature, input_example)
        
        # Create and log summary report
        create_summary_report(data_stats, metrics, feature_metadata)
        
        
        print("\nTraining completed successfully!")
        print(f"MLflow run ID: {run.info.run_id}")
        
        return preprocessor, ensemble, metrics

def evaluate_ensemble(ensemble, X, y):
    """Evaluate ensemble model with comprehensive metrics"""
    # Get predictions from each model
    xgb_probs = ensemble.xgb_model.predict_proba(X)[:, 1]
    rf_probs = ensemble.rf_model.predict_proba(X)[:, 1]
    nn_probs = ensemble.nn_model(torch.FloatTensor(X)).detach().numpy()[:, 1]
    
    # Calculate weighted ensemble predictions
    ensemble_probs = (
        ensemble.weights[0] * xgb_probs +
        ensemble.weights[1] * rf_probs +
        ensemble.weights[2] * nn_probs
    )
    
    # Calculate threshold-based predictions
    ensemble_preds = (ensemble_probs >= 0.5).astype(int)
    
    # Calculate metrics
    metrics = {
        'auc': roc_auc_score(y, ensemble_probs),
        'accuracy': accuracy_score(y, ensemble_preds),
        'precision': precision_score(y, ensemble_preds),
        'recall': recall_score(y, ensemble_preds),
        'f1': f1_score(y, ensemble_preds),
        'confusion_matrix': confusion_matrix(y, ensemble_preds).tolist()
    }
    
    # Calculate individual model metrics
    model_probs = {
        'xgboost': xgb_probs,
        'random_forest': rf_probs,
        'neural_network': nn_probs
    }
    
    for model_name, probs in model_probs.items():
        preds = (probs >= 0.5).astype(int)
        metrics[f'{model_name}_auc'] = roc_auc_score(y, probs)
        metrics[f'{model_name}_precision'] = precision_score(y, preds)
        metrics[f'{model_name}_recall'] = recall_score(y, preds)
        metrics[f'{model_name}_f1'] = f1_score(y, preds)
        metrics[f'{model_name}_accuracy'] = accuracy_score(y, preds)
    
    return metrics

def log_models(ensemble, signature, input_example):
    """Log all models with proper signatures and examples"""
    # Log XGBoost model
    mlflow.sklearn.log_model(
        ensemble.xgb_model,
        "xgboost_model",
        signature=signature,
        input_example=input_example,
        registered_model_name="fd_xgboost"
    )
    
    # Log Random Forest model
    mlflow.sklearn.log_model(
        ensemble.rf_model,
        "random_forest_model",
        signature=signature,
        input_example=input_example,
        registered_model_name="fd_random_forest"
    )
    
    # Log Neural Network model with custom wrapper
    wrapped_nn = FraudDetectionNNWrapper(ensemble.nn_model)
    nn_signature = mlflow.models.signature.infer_signature(
        input_example, 
        wrapped_nn.predict(None, input_example)
    )
    
    mlflow.pyfunc.log_model(
        "neural_network_model",
        python_model=wrapped_nn,
        signature=nn_signature,
        input_example=input_example,
        registered_model_name="fd_torch"
    )

def create_summary_report(data_stats, metrics, feature_metadata):
    """Create comprehensive training summary report"""
    summary_report = f"""
    Fraud Detection Training Summary
    ==============================
    
    Dataset Statistics:
    - Total samples: {data_stats['total_samples']:,}
    - Training samples: {data_stats['training_samples']:,}
    - Validation samples: {data_stats['validation_samples']:,}
    - Test samples: {data_stats['test_samples']:,}
    - Feature dimension: {data_stats['feature_dimension']}
    - Original fraud ratio: {data_stats['original_fraud_ratio']:.3%}
    - Training fraud ratio: {data_stats['training_fraud_ratio']:.3%}
    - Validation fraud ratio: {data_stats['validation_fraud_ratio']:.3%}
    - Test fraud ratio: {data_stats['test_fraud_ratio']:.3%}
    
    Model Performance:
    - XGBoost AUC: {metrics['test']['xgboost_auc']:.3f}
    - Random Forest AUC: {metrics['test']['random_forest_auc']:.3f}
    - Neural Network AUC: {metrics['test']['neural_network_auc']:.3f}
    - Ensemble AUC: {metrics['test']['auc']:.3f}
    - XGBoost Recall: {metrics['test']['xgboost_recall']:.3f}
    - Random Forest Recall: {metrics['test']['random_forest_recall']:.3f}
    - Neural Network Recall: {metrics['test']['neural_network_recall']:.3f}
    - Ensemble Recall: {metrics['test']['recall']:.3f}
    - XGBoost Precsion: {metrics['test']['xgboost_precision']:.3f}
    - Random Forest Precision: {metrics['test']['random_forest_precision']:.3f}
    - Neural Network Precision: {metrics['test']['neural_network_precision']:.3f}
    - Ensemble Precision: {metrics['test']['precision']:.3f}
    - XGBoost Accuracy: {metrics['test']['xgboost_accuracy']:.3f}
    - Random Forest Accuracy: {metrics['test']['random_forest_accuracy']:.3f}
    - Neural Network Accuracy: {metrics['test']['neural_network_accuracy']:.3f}
    - Ensemble Accuracy: {metrics['test']['accuracy']:.3f}
    - XGBoost F1: {metrics['test']['xgboost_f1']:.3f}
    - Random Forest F1: {metrics['test']['random_forest_f1']:.3f}
    - Neural Network F1: {metrics['test']['neural_network_f1']:.3f}
    - Ensemble F1: {metrics['test']['f1']:.3f}"
    
    Feature Groups:
    {'-' * 40}
    """
    
    for group, metadata in feature_metadata.items():
        summary_report += f"\n- {group}: {metadata['feature_count']} features"
        summary_report += f"\n  Scaler: {metadata['scaler_type']}"
        summary_report += f"\n  Imputation: {metadata['imputer_strategy']}"
    
    with open("training_summary.txt", "w") as f:
        f.write(summary_report)
    mlflow.log_artifact("training_summary.txt")# Define the directory containing the files


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
pos_df = combined_df[combined_df['TX_FRAUD'] == 1].iloc[:100]
neg_df = combined_df[combined_df['TX_FRAUD'] == 0].iloc[:10000]
df2 = pd.concat([pos_df, neg_df], ignore_index=True, axis= 0)
preprocessor, ensemble, metrics = train_fraud_detection_system(combined_df)