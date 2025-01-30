import sys
import os
from src.preprocessing.prep import get_train_test_set, is_weekend, is_night, get_customer_spending_features, get_count_risk_rolling_window
import pandas as pd
import numpy as np
import datetime 
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, precision_score, accuracy_score, recall_score, f1_score, average_precision_score,precision_recall_curve
import xgboost as xgb
import mlflow
import json
import warnings
import itertools
warnings.filterwarnings("ignore")

class FraudDetectionXGBoost:
    def __init__(self, input_dim=None, feature_names=None, param_config=dbutils.widgets.get('PARAM_CONFIG')):
        self.input_dim = input_dim
        self.feature_names = feature_names
        self.threshold = 0.5
        with open(param_config, 'r') as file:
            self.param_space = json.load(file)
        self.xgb_model = None
        self.calibrated_model = None
        self.experiment_name = dbutils.widgets.get('MLFLOW_DIR')
        
    def _calculate_metrics(self, y_true, y_prob, threshold=None):
        """
        Calculate comprehensive metrics for model evaluation
        """
        if threshold is None:
            threshold = self.threshold
            
        y_pred = (y_prob >= threshold).astype(int)
        
        return {
            'average_precision': average_precision_score(y_true, y_prob),
            'auc': roc_auc_score(y_true, y_prob),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'accuracy': accuracy_score(y_true, y_pred)
        }

    def _optimize_threshold(self, y_true, y_prob):
        """
        Optimize threshold using F2 score with minimum recall constraint
        """
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
        thresholds = np.append(thresholds, 1.0)
        
        valid_points = recalls >= 0.70
        
        if not np.any(valid_points):
            print("Warning: No threshold achieves minimum recall of 0.70")
            # Find threshold that maximizes F2 score without constraint
            f2_scores = (5 * precisions * recalls) / (4 * precisions + recalls)
            best_idx = np.argmax(f2_scores)
        else:
            # Among points with sufficient recall, maximize F2 score
            valid_precisions = precisions[valid_points]
            valid_recalls = recalls[valid_points]
            valid_thresholds = thresholds[valid_points]
            
            f2_scores = (5 * valid_precisions * valid_recalls) / (4 * valid_precisions + valid_recalls)
            best_idx = np.argmax(f2_scores)
            
        selected_threshold = thresholds[best_idx]
        metrics = self._calculate_metrics(y_true, y_prob, selected_threshold)
        
        print(f"\nAt selected threshold {selected_threshold:.4f}:")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"F2 Score: {(5 * metrics['precision'] * metrics['recall']) / (4 * metrics['precision'] + metrics['recall']):.4f}")    
        return selected_threshold

    def _generate_param_combinations(self):
        param_names = list(self.param_space.keys())
        param_values = list(self.param_space.values())
        return [dict(zip(param_names, combo)) for combo in itertools.product(*param_values)]

    def _tune_xgboost(self, X_train, y_train, X_val, y_val):
        print("\nTuning XGBoost hyperparameters...")
        
        with mlflow.start_run(nested=True, run_name="xgboost_tuning", 
                            experiment_id=mlflow.get_experiment_by_name(self.experiment_name).experiment_id):
            best_score = -float('inf')
            best_params = None
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
                        eval_metric=['aucpr']
                    )
                    
                    model.fit(
                        X_cv_train, y_cv_train,
                        eval_set=[(X_cv_val, y_cv_val)],
                        early_stopping_rounds=30,
                        verbose=False
                    )
                    
                    # Get calibrated probabilities
                    calibrated = CalibratedClassifierCV(model, cv=5, method='sigmoid')
                    calibrated.fit(X_cv_train, y_cv_train)
                    y_prob = calibrated.predict_proba(X_cv_val)[:, 1]
                    
                    # Calculate average precision score
                    cv_scores.append(average_precision_score(y_cv_val, y_prob))
                
                mean_score = np.mean(cv_scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = params
            
            mlflow.log_params(best_params)
            print("\nBest parameters:")
            for param, value in best_params.items():
                print(f"{param}: {value}")
            
            self.xgb_model = xgb.XGBClassifier(
                **best_params,
                tree_method='hist',
                eval_metric=['aucpr']
            )

    def train(self, X_train, y_train, X_val, y_val):
        print("\nTraining XGBoost model...")
        self._tune_xgboost(X_train, y_train, X_val, y_val)
        
        # Train final model
        self.xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=30,
            verbose=False
        )
        
        # Train calibrated model
        print("\nCalibrating probabilities...")
        self.calibrated_model = CalibratedClassifierCV(self.xgb_model, cv=5, method='sigmoid')
        self.calibrated_model.fit(X_train, y_train)
        
        # Optimize threshold on validation set using calibrated probabilities
        y_val_prob = self.calibrated_model.predict_proba(X_val)[:, 1]
        self.threshold = self._optimize_threshold(y_val, y_val_prob)
        print(f"\nOptimized threshold: {self.threshold:.4f}")

    def predict_proba(self, X):
        """
        Get calibrated probability predictions
        """
        return self.calibrated_model.predict_proba(X)[:, 1]

    def predict(self, X):
        """
        Get binary predictions using optimized threshold
        """
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def evaluate(self, X, y):
        """
        Evaluate model performance using calibrated probabilities
        """
        y_prob = self.predict_proba(X)
        return self._calculate_metrics(y, y_prob)
    
def train_fraud_detection_system(raw_data):
    print("Starting fraud detection system training...")
    train_df, test_df = get_train_test_set(raw_data, 
                                          start_date_training=raw_data['TX_DATETIME'].min(), 
                                          delta_train=7, 
                                          delta_delay=3, 
                                          delta_test=7)
    
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

    # Calculate and print data statistics
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
        # Log data statistics
        mlflow.log_params(data_stats)

        # Train model with new implementation
        model.train(X_train, y_train, X_val, y_val)

        # Evaluate using calibrated predictions
        print("\nEvaluating on test data...")
        test_metrics = model.evaluate(X_test, y_test)

        print("\nTest Metrics:")
        for metric, value in test_metrics.items():
            print(f"{metric}: {value:.4f}")

        mlflow.log_metrics(test_metrics)

        # Log both the base model and calibrated model
        input_example = pd.DataFrame(X_train[:1], columns=feature_names)
        
        # Log base XGBoost model
        base_signature = mlflow.models.signature.infer_signature(
            X_train, 
            model.xgb_model.predict_proba(X_train)
        )
        mlflow.sklearn.log_model(
            sk_model=model.xgb_model,
            artifact_path="xgboost_base_model",
            input_example=input_example,
            signature=base_signature,
            registered_model_name="fraud_detection_xgboost_base"
        )
        
        # Log calibrated model
        calibrated_signature = mlflow.models.signature.infer_signature(
            X_train, 
            model.calibrated_model.predict_proba(X_train)
        )
        mlflow.sklearn.log_model(
            sk_model=model.calibrated_model,
            artifact_path="xgboost_calibrated_model",
            input_example=input_example,
            signature=calibrated_signature,
            registered_model_name="fraud_detection_xgboost_calibrated"
        )
        
        # Log threshold
        mlflow.log_param("optimal_threshold", model.threshold)

    print("\nXGBoost training and evaluation completed successfully.")

# Keep the data loading code exactly as is
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

# Start training
train_fraud_detection_system(df)