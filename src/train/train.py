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
import itertools
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
        # Original initialization
        self.input_dim = input_dim
        self.feature_names = feature_names or [f'feature_{i}' for i in range(input_dim)]
        self.weights = weights
        
        # Initialize class weights based on original distribution
        weight_ratio = (1/0.026)  # Based on original class distribution
        self.class_weights = {0: 1, 1: weight_ratio}
        
        # Initialize thresholds
        self.thresholds = {
            'xgboost': 0.3,
            'random_forest': 0.3,
            'neural_network': 0.3,
            'ensemble': 0.3
        }
        
        # Parameter spaces for tuning
        self.param_space = {
            'xgboost': {
                'max_depth': [3, 4, 5],
                'learning_rate': [0.01, 0.05],
                'n_estimators': [300, 500],
                'min_child_weight': [5, 10],
                'subsample': [0.7, 0.8],
                'colsample_bytree': [0.7, 0.8],
                'gamma': [0.5, 1],
                'reg_alpha': [0.5, 1],
                'reg_lambda': [1, 2],
                'scale_pos_weight': [weight_ratio] 
            },
            'random_forest': {
                'n_estimators': [200, 300],
                'max_depth': [4, 6],
                'min_samples_leaf': [2, 4],
                'min_samples_split': [5, 10]
            },
            'neural_network': {
                'learning_rate': [0.0005, 0.001],
                'dropout_rates': [(0.4, 0.3, 0.2), (0.5, 0.4, 0.3)],
                'hidden_layers': [(64, 32, 16), (32, 16, 8)],
                'weight_decay': [1e-5, 1e-4]
            }
        }
        
        # Initialize models with default parameters
        self.xgb_model = None
        self.rf_model = None
        self.nn_model = None

    def _optimize_threshold(self, y_true, y_prob, metric='f1', threshold_range=None):
        """Original threshold optimization logic"""
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
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        return best_threshold, best_score

    def _optimize_model_thresholds(self, X_val, y_val, metric='f1'):
        """Optimize classification thresholds for all models"""
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
    
    def _optimize_weights(self, X_val, y_val):
        """Original weight optimization logic"""
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
            
            # Optimize for both AUC and recall
            auc_score = roc_auc_score(y_val, ensemble_probs)
            optimal_threshold, _ = self._optimize_threshold(y_val, ensemble_probs, 'f1')
            ensemble_preds = (ensemble_probs >= optimal_threshold).astype(int)
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
        mlflow.log_params({
            'xgboost_weight': self.weights[0],
            'rf_weight': self.weights[1],
            'nn_weight': self.weights[2]
        })
        
        print(f"Optimized weights: XGBoost={self.weights[0]:.3f}, "
              f"RF={self.weights[1]:.3f}, NN={self.weights[2]:.3f}")
        
        return self.weights

    def _tune_xgboost(self, X_train, y_train, X_val, y_val):
        """Tune XGBoost with MLFlow tracking"""
        print("\nTuning XGBoost hyperparameters...")
        
        with mlflow.start_run(nested=True, run_name="xgboost_tuning"):
            best_score = -float('inf')
            best_params = None
            best_threshold = 0.3
            
            tscv = TimeSeriesSplit(n_splits=3)
            
            for params in self._generate_param_combinations('xgboost'):
                cv_scores = []
                cv_thresholds = []
                
                for train_idx, val_idx in tscv.split(X_train):
                    X_cv_train, X_cv_val = X_train[train_idx], X_train[val_idx]
                    y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]
                    
                    model = xgb.XGBClassifier(
                        **params,
                        tree_method='hist',
                        eval_metric=['auc', 'aucpr']
                    )
                    
                    model.fit(
                        X_cv_train, y_cv_train,
                        eval_set=[(X_cv_val, y_cv_val)],
                        early_stopping_rounds=30,
                        verbose=False
                    )
                    
                    y_pred_proba = model.predict_proba(X_cv_val)[:, 1]
                    threshold, score = self._optimize_threshold(y_cv_val, y_pred_proba, 'f1')
                    
                    cv_scores.append(score)
                    cv_thresholds.append(threshold)
                
                mean_score = np.mean(cv_scores)
                mean_threshold = np.mean(cv_thresholds)
                
                mlflow.log_metrics({
                    'xgb_cv_score': mean_score,
                    'xgb_cv_threshold': mean_threshold
                })
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = params
                    best_threshold = mean_threshold
            
            mlflow.log_params({f"xgb_{k}": v for k, v in best_params.items()})
            self.thresholds['xgboost'] = best_threshold
            
            # Initialize final model with best parameters
            self.xgb_model = xgb.XGBClassifier(**best_params, tree_method='hist')
            
            return best_params

    def _tune_random_forest(self, X_train, y_train, X_val, y_val):
        """Tune Random Forest with MLFlow tracking"""
        print("\nTuning Random Forest hyperparameters...")
        
        with mlflow.start_run(nested=True, run_name="random_forest_tuning"):
            best_score = -float('inf')
            best_params = None
            best_threshold = 0.3
            
            tscv = TimeSeriesSplit(n_splits=3)
            
            for params in self._generate_param_combinations('random_forest'):
                cv_scores = []
                cv_thresholds = []
                
                for train_idx, val_idx in tscv.split(X_train):
                    X_cv_train, X_cv_val = X_train[train_idx], X_train[val_idx]
                    y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]
                    
                    model = RandomForestClassifier(
                        **params,
                        class_weight='balanced_subsample',
                        n_jobs=-1,
                        random_state=42
                    )
                    
                    model.fit(X_cv_train, y_cv_train)
                    y_pred_proba = model.predict_proba(X_cv_val)[:, 1]
                    threshold, score = self._optimize_threshold(y_cv_val, y_pred_proba, 'recall')
                    
                    cv_scores.append(score)
                    cv_thresholds.append(threshold)
                
                mean_score = np.mean(cv_scores)
                mean_threshold = np.mean(cv_thresholds)
                
                mlflow.log_metrics({
                    'rf_cv_score': mean_score,
                    'rf_cv_threshold': mean_threshold
                })
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = params
                    best_threshold = mean_threshold
            
            mlflow.log_params({f"rf_{k}": v for k, v in best_params.items()})
            self.thresholds['random_forest'] = best_threshold
            
            # Initialize final model with best parameters
            self.rf_model = RandomForestClassifier(
                **best_params,
                class_weight='balanced_subsample',
                n_jobs=-1,
                random_state=42
            )
            
            return best_params

    def _tune_neural_network(self, X_train, y_train, X_val, y_val):
        """Tune Neural Network with MLFlow tracking"""
        print("\nTuning Neural Network hyperparameters...")
        
        with mlflow.start_run(nested=True, run_name="neural_network_tuning"):
            best_score = -float('inf')
            best_params = None
            best_threshold = 0.3
            best_state_dict = None
            
            tscv = TimeSeriesSplit(n_splits=3)
            
            for params in self._generate_param_combinations('neural_network'):
                cv_scores = []
                cv_thresholds = []
                
                for train_idx, val_idx in tscv.split(X_train):
                    X_cv_train, X_cv_val = X_train[train_idx], X_train[val_idx]
                    y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]
                    
                    model = self._create_nn_model(params)
                    X_cv_train_tensor = torch.FloatTensor(X_cv_train)
                    y_cv_train_tensor = torch.LongTensor(y_cv_train)
                    X_cv_val_tensor = torch.FloatTensor(X_cv_val)
                    
                    criterion = nn.CrossEntropyLoss(
                        weight=torch.FloatTensor([1, self.class_weights[1]])
                    )
                    optimizer = torch.optim.Adam(
                        model.parameters(),
                        lr=params['learning_rate'],
                        weight_decay=params['weight_decay']
                    )
                    
                    patience = 10
                    patience_counter = 0
                    best_val_score = -float('inf')
                    best_local_state = None
                    
                    for epoch in range(100):
                        model.train()
                        optimizer.zero_grad()
                        outputs = model(X_cv_train_tensor)
                        loss = criterion(outputs, y_cv_train_tensor)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        
                        model.eval()
                        with torch.no_grad():
                            val_outputs = model(X_cv_val_tensor)
                            val_probs = val_outputs.numpy()[:, 1]
                            threshold, score = self._optimize_threshold(y_cv_val, val_probs, 'recall')
                            
                            if score > best_val_score:
                                best_val_score = score
                                patience_counter = 0
                                best_local_state = model.state_dict()
                            else:
                                patience_counter += 1
                            
                            if patience_counter >= patience:
                                break
                    
                    model.load_state_dict(best_local_state)
                    cv_scores.append(best_val_score)
                    cv_thresholds.append(threshold)
                
                mean_score = np.mean(cv_scores)
                mean_threshold = np.mean(cv_thresholds)
                
                mlflow.log_metrics({
                    'nn_cv_score': mean_score,
                    'nn_cv_threshold': mean_threshold
                })
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = params
                    best_threshold = mean_threshold
                    best_state_dict = best_local_state
            
            mlflow.log_params({f"nn_{k}": str(v) for k, v in best_params.items()})
            self.thresholds['neural_network'] = best_threshold
            
            # Initialize final model with best parameters
            self.nn_model = self._create_nn_model(best_params)
            self.nn_model.load_state_dict(best_state_dict)
            
            return best_params

    def _create_nn_model(self, params):
        """Create Neural Network model with given parameters"""
        class CustomNN(nn.Module):
            def __init__(self, input_dim, hidden_layers, dropout_rates):
                super().__init__()
                
                layers = []
                prev_dim = input_dim
                
                for hidden_dim, dropout_rate in zip(hidden_layers, dropout_rates):
                    layers.extend([
                        nn.Linear(prev_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout_rate)
                    ])
                    prev_dim = hidden_dim
                
                layers.append(nn.Linear(prev_dim, 2))
                self.model = nn.Sequential(*layers)
                
                # Initialize weights
                for m in self.model:
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_normal_(m.weight)
                        nn.init.constant_(m.bias, 0)
            
            def forward(self, x):
                logits = self.model(x)
                return nn.functional.softmax(logits, dim=1)
        
        return CustomNN(
            self.input_dim,
            params['hidden_layers'],
            params['dropout_rates']
        )

    def _generate_param_combinations(self, model_type):
        """Generate parameter combinations for grid search"""
        param_space = self.param_space[model_type]
        param_names = list(param_space.keys())
        param_values = list(param_space.values())
        
        combinations = list(itertools.product(*param_values))
        return [dict(zip(param_names, combo)) for combo in combinations]

    def train(self, X_train, y_train, X_val, y_val):
        """Enhanced training with hyperparameter tuning and original optimization"""
        print(f"Starting ensemble training with hyperparameter tuning...")
        
        with mlflow.start_run(nested=True, run_name="ensemble_tuning") as run:
            # Log initial setup
            mlflow.log_params({
                'input_dim': self.input_dim,
                'initial_xgb_weight': self.weights[0],
                'initial_rf_weight': self.weights[1],
                'initial_nn_weight': self.weights[2]
            })
            
            # Tune individual models
            print("\nTuning individual models...")
            xgb_params = self._tune_xgboost(X_train, y_train, X_val, y_val)
            rf_params = self._tune_random_forest(X_train, y_train, X_val, y_val)
            nn_params = self._tune_neural_network(X_train, y_train, X_val, y_val)
            
            # Train final models with best parameters
            print("\nTraining final models with best parameters...")
            
            # XGBoost
            self.xgb_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=30,
                verbose=100
            )
            
            # Random Forest
            self.rf_model.fit(X_train, y_train)
            
            # Neural Network (already trained in tuning)
            self.nn_model.eval()
            
            # Optimize ensemble weights
            print("\nOptimizing ensemble weights...")
            optimized_weights = self._optimize_weights(X_val, y_val)
            
            # Optimize final thresholds
            print("\nOptimizing final thresholds...")
            self._optimize_model_thresholds(X_val, y_val, metric='f1')
            
            # Get validation predictions for final evaluation
            val_metrics = self._evaluate_ensemble(X_val, y_val)
            
            # Log final metrics
            mlflow.log_metrics({
                'final_val_auc': val_metrics['ensemble_auc'],
                'final_val_precision': val_metrics['ensemble_precision'],
                'final_val_recall': val_metrics['ensemble_recall'],
                'final_val_f1': val_metrics['ensemble_f1']
            })
            
            # Log model parameters
            final_params = {
                'xgboost': xgb_params,
                'random_forest': rf_params,
                'neural_network': nn_params,
                'ensemble_weights': optimized_weights,
                'thresholds': self.thresholds
            }
            
            mlflow.log_dict(final_params, 'model_parameters.json')
            
            print("\nTraining completed successfully!")
            return val_metrics

    def _evaluate_ensemble(self, X, y):
        """Evaluate ensemble performance"""
        # Get predictions from each model
        xgb_probs = self.xgb_model.predict_proba(X)[:, 1]
        rf_probs = self.rf_model.predict_proba(X)[:, 1]
        nn_probs = self.nn_model(torch.FloatTensor(X)).detach().numpy()[:, 1]
        
        # Individual model metrics
        metrics = {}
        for name, probs in [
            ('xgboost', xgb_probs),
            ('random_forest', rf_probs),
            ('neural_network', nn_probs)
        ]:
            preds = (probs >= self.thresholds[name]).astype(int)
            metrics.update({
                f'{name}_auc': roc_auc_score(y, probs),
                f'{name}_precision': precision_score(y, preds),
                f'{name}_recall': recall_score(y, preds),
                f'{name}_f1': f1_score(y, preds),
                f'{name}_accuracy': accuracy_score(y, preds)
            })
        
        # Ensemble predictions
        ensemble_probs = (
            self.weights[0] * xgb_probs +
            self.weights[1] * rf_probs +
            self.weights[2] * nn_probs
        )
        
        ensemble_preds = (ensemble_probs >= self.thresholds['ensemble']).astype(int)
        
        # Ensemble metrics
        metrics.update({
            'ensemble_auc': roc_auc_score(y, ensemble_probs),
            'ensemble_precision': precision_score(y, ensemble_preds),
            'ensemble_recall': recall_score(y, ensemble_preds),
            'ensemble_f1': f1_score(y, ensemble_preds),
            'ensemble_accuracy' : accuracy_score(y, ensemble_preds),
            'confusion_matrix': confusion_matrix(y, ensemble_preds).tolist()
        })
        
        return metrics

    def predict_proba(self, X):
        """Get probability predictions with optimized weights"""
        X_tensor = torch.FloatTensor(X)
        
        # Get predictions from each model
        xgb_probs = self.xgb_model.predict_proba(X)[:, 1]
        rf_probs = self.rf_model.predict_proba(X)[:, 1]
        nn_probs = self.nn_model(X_tensor).detach().numpy()[:, 1]
        
        # Calculate weighted probabilities
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

    def get_feature_importance(self, method='ensemble'):
        """Get feature importance with support for tuned models"""
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
            
            # Weighted average using tuned weights
            ensemble_importance = (
                self.weights[0] * xgb_importance +
                self.weights[1] * rf_importance +
                self.weights[2] * nn_importance
            )
            
            return {
                name: score for name, score in zip(self.feature_names, ensemble_importance)
            }
        else:
            raise ValueError(f"Unsupported importance method: {method}")

    def save_model(self, path):
        """Save the tuned ensemble model"""
        model_data = {
            'weights': self.weights,
            'thresholds': self.thresholds,
            'feature_names': self.feature_names,
            'xgb_model': self.xgb_model,
            'rf_model': self.rf_model,
            'nn_model_state': self.nn_model.state_dict(),
            'input_dim': self.input_dim,
            'best_params': {
                'xgboost': self.xgb_model.get_params(),
                'random_forest': self.rf_model.get_params(),
                'neural_network': {
                    k: v for k, v in self.nn_model.state_dict().items()
                }
            }
        }
        torch.save(model_data, path)

    @classmethod
    def load_model(cls, path):
        """Load a tuned ensemble model"""
        model_data = torch.load(path)
        
        # Initialize ensemble
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