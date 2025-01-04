import sys
import os
from src.preprocessing.prepp import get_train_test_set, is_weekend, is_night, get_customer_spending_features, get_count_risk_rolling_window
import pandas as pd
import numpy as np
import datetime 
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, precision_score, accuracy_score, recall_score, f1_score, confusion_matrix, precision_recall_curve, auc
import xgboost as xgb
import mlflow
import matplotlib.pyplot as plt
import json
import warnings
import itertools
from scipy.optimize import minimize
warnings.filterwarnings("ignore")

class FraudDetectionXGBoost:
    def __init__(self, input_dim = None, feature_names = None, param_config = dbutils.widgets.get('PARAM_CONFIG')):
        self.input_dim = input_dim
        self.feature_names = feature_names 
        self.threshold = 0.5
        with open(param_config, 'r') as file:
            self.param_space = json.load(file)       
        self.xgb_model = None
        self.experiment_name = dbutils.widgets.get('MLFLOW_DIR')

    def _calculate_class_weight(self, y):
        """
        Calculate class weights based on class distribution
        """
        class_counts = np.bincount(y)
        total_samples = len(y)
        weights = {
            0: total_samples / (2 * class_counts[0]),
            1: total_samples / (2 * class_counts[1])
        }
        return weights
    
    def _optimize_threshold(self, y_true, y_prob):
        """
        Optimize threshold to maximize recall while maintaining precision >= 80%
        """
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
        # Add the endpoint (1.0) since precision_recall_curve doesn't include it
        thresholds = np.append(thresholds, 1.0)
        
        valid_points = precisions >= 0.75       
        if not np.any(valid_points):
            print("Warning: No threshold achieves base precision. Using highest precision threshold.")
            best_idx = np.argmax(precisions)
        else:
            # Among points with sufficient precision, maximize recall
            valid_recalls = recalls[valid_points]
            valid_thresholds = thresholds[valid_points]
            
            best_idx = np.argmax(valid_recalls)
            threshold = valid_thresholds[best_idx]

            print(f"\nAt selected threshold:")
            print(f"Precision: {precisions[valid_points][best_idx]:.4f}")
            print(f"Recall: {valid_recalls[best_idx]:.4f}")
            
            return threshold
            
        return thresholds[best_idx]

    def _generate_param_combinations(self):
        param_names = list(self.param_space.keys())
        param_values = list(self.param_space.values())
        return [dict(zip(param_names, combo)) for combo in itertools.product(*param_values)]

    def _tune_xgboost(self, X_train, y_train, X_val, y_val):
        print("\nTuning XGBoost hyperparameters...")

        # Calculate class weights
        class_weights = self._calculate_class_weight(y_train)
        sample_weights = np.array([class_weights[y] for y in y_train])

        with mlflow.start_run(nested=True, run_name="xgboost_tuning", experiment_id=
                              mlflow.get_experiment_by_name(self.experiment_name).experiment_id):
            best_score = -float('inf')
            best_params = None
            best_threshold = 0.5
            tscv = TimeSeriesSplit(n_splits=5)
            for params in self._generate_param_combinations():
                cv_scores = []
                for train_idx, val_idx in tscv.split(X_train):
                    X_cv_train, X_cv_val = X_train[train_idx], X_train[val_idx]
                    y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]

                    if len(np.unique(y_cv_val)) < 2:
                        continue
                    model = xgb.XGBClassifier(
                        **params,
                        tree_method='hist',
                        eval_metric=['aucpr'],
                        scale_pos_weight = class_weights[1]/class_weights[0]
                    )
                    model.fit(
                        X_cv_train, y_cv_train,
                        sample_weight=sample_weights,
                        eval_set=[(X_cv_val, y_cv_val)],
                        early_stopping_rounds=30,
                        verbose=False
                    )
                    y_pred_proba = model.predict_proba(X_cv_val)[:, 1]

                    # Find optimal threshold for this fold
                    precisions, recalls, thresholds = precision_recall_curve(y_cv_val, y_pred_proba)
                    valid_points = precisions >= 0.75
                    if np.any(valid_points):
                        valid_recalls = recalls[valid_points]
                        max_recall = np.max(valid_recalls)
                        cv_scores.append(max_recall)
                    else:
                        cv_scores.append(0.0)  # Penalize if can't achieve desired precision
                
                mean_score = np.mean(cv_scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = params
            
            mlflow.log_params(best_params)
            # Initialize model with best parameters and class weights
            self.xgb_model = xgb.XGBClassifier(
                **best_params,
                tree_method='hist',
                scale_pos_weight=class_weights[1]/class_weights[0]
            )

    def train(self, X_train, y_train, X_val, y_val):
        print("\nTraining XGBoost model...")
        self._tune_xgboost(X_train, y_train, X_val, y_val)

        class_weights = self._calculate_class_weight(y_train)
        sample_weights = np.array([class_weights[y] for y in y_train])

        self.xgb_model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=30,
            verbose=False
        )
        #optimize threshold on validation set
        y_val_pred = self.xgb_model.predict_proba(X_val)[:, 1]
        self.threshold = self._optimize_threshold(y_val, y_val_pred)
        print(f"\nOptimized threshold: {self.threshold:.4f}")

    def evaluate(self, X, y):
        y_prob = self.xgb_model.predict_proba(X)[:, 1]
        y_pred = (y_prob >= self.threshold).astype(int)
        return {
            'auc': roc_auc_score(y, y_prob),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred),
            'accuracy': accuracy_score(y, y_pred)
        }

def train_fraud_detection_system(raw_data):
    print("Starting fraud detection system training...")
    train_df, test_df = get_train_test_set(raw_data, start_date_training=raw_data['TX_DATETIME'].min(), delta_train=7, delta_delay=3, delta_test=7)
    
    def prep_data(df):
        df['TX_DURING_WEEKEND'] = df['TX_DATETIME'].apply(is_weekend)
        df['TX_DURING_NIGHT'] = df['TX_DATETIME'].apply(is_night)
        df = df.groupby('CUSTOMER_ID').apply(lambda x: get_customer_spending_features(x, windows_size_in_days=[1, 7, 30]))
        df = df.sort_values('TX_DATETIME').reset_index(drop=True)
        df = df.groupby('TERMINAL_ID').apply(lambda x: get_count_risk_rolling_window(x, delay_period=7, windows_size_in_days=[1, 7, 30], feature="TERMINAL_ID"))
        df = df.sort_values('TX_DATETIME').reset_index(drop=True)
        features = df.drop(['TX_DATETIME', 'TX_FRAUD', 'TX_FRAUD_SCENARIO'], axis=1).columns.to_list()
        X = df.drop(['TX_DATETIME', 'TX_FRAUD', 'TX_FRAUD_SCENARIO'], axis=1).to_numpy()
        y = df['TX_FRAUD'].to_numpy()
        return X, y, features

    print("Preprocessing training data...")
    X_train, y_train, train_features = prep_data(train_df)

    print("Preprocessing validation data...")
    grouped = test_df.groupby('TX_FRAUD')
    test_df_final, val_df = [], []
    for _, group in grouped:
        mid = len(group) // 2
        test_df_final.append(group.iloc[:mid])
        val_df.append(group.iloc[mid:])
    test_df_final = pd.concat(test_df_final, axis=0).reset_index(drop=True)
    val_df = pd.concat(val_df, axis=0).reset_index(drop=True)
    X_val, y_val, val_features = prep_data(val_df)

    print("Preprocessing test data...")
    X_test, y_test, test_features = prep_data(test_df_final)

    print("\nData Statistics:")
    data_stats = {
        'total_samples': len(raw_data),
        'training_samples': len(X_train),
        'validation_samples': len(X_val),
        'test_samples': len(X_test),
        'feature_dimension': X_train.shape[1],
        'original_fraud_ratio': float(raw_data['TX_FRAUD'].mean()),
        'training_fraud_ratio': float(y_train.mean()),
        'validation_fraud_ratio': float(y_val.mean()),
        'test_fraud_ratio': float(y_test.mean())
    }
    for stat, value in data_stats.items():
        print(f"{stat}: {value}")

    print("\nInitializing XGBoost model...")
    feature_names = train_features
    model = FraudDetectionXGBoost(
        input_dim=X_train.shape[1],
        feature_names=feature_names
    )

    exp = dbutils.widgets.get('MLFLOW_DIR')
    with mlflow.start_run(run_name="fraud_detection_xgboost", 
                          experiment_id=mlflow.get_experiment_by_name(exp).experiment_id):
        mlflow.log_params(data_stats)

        model.train(X_train, y_train, X_val, y_val)

        print("\nEvaluating on test data...")
        test_metrics = model.evaluate(X_test, y_test)

        print("\nTest Metrics:")
        for metric, value in test_metrics.items():
            print(f"{metric}: {value:.4f}")

        mlflow.log_metrics(test_metrics)

        # Log model and artifacts
        input_example = pd.DataFrame(X_train[:1], columns=feature_names)
        signature = mlflow.models.signature.infer_signature(X_train, model.xgb_model.predict_proba(X_train))
        mlflow.sklearn.log_model(
            sk_model=model.xgb_model,
            artifact_path="xgboost_model",
            input_example=input_example,
            signature=signature,
            registered_model_name="fraud_detection_xgboost"
        )

    print("\nXGBoost training and evaluation completed successfully.")

# Load data and trigger the training system

directory = dbutils.widgets.get('DIR_NAME')
df_list = []
for file in os.listdir(directory):
    if file.endswith('.csv'):
        file_path = os.path.join(directory, file)
        df = pd.read_csv(file_path)
        df_list.append(df)

# Concatenate all DataFrames into a single DataFrame
df = pd.concat(df_list, ignore_index=True)
df = df.drop(df.columns[0], axis=1)
train_fraud_detection_system(df)