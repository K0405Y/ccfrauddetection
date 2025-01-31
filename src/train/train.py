import sys
import os
from src.preprocessing.prep import get_train_test_set, is_weekend, is_night, get_customer_spending_features, get_count_risk_rolling_window
import pandas as pd
import numpy as np
import datetime 
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_score, accuracy_score, recall_score, f1_score, average_precision_score,precision_recall_curve, auc
import xgboost as xgb
import mlflow
import matplotlib.pyplot as plt
import json
import warnings
import itertools
warnings.filterwarnings("ignore")

class FraudDetectionXGBoost:
    """
    XGBoost-based fraud detection model with threshold optimization and MLflow tracking.
    """

    def __init__(self, input_dim=None, feature_names=None, param_config=dbutils.widgets.get('PARAM_CONFIG')):
        """
        Initializes the fraud detection model.

        Parameters:
        - input_dim: Number of input features (default: None, set later)
        - feature_names: List of feature names (default: None)
        - param_config: Path to JSON file containing hyperparameter search space (set via Databricks widget)
        """
        self.input_dim = input_dim  # Stores the number of input features
        self.feature_names = feature_names  # Stores the names of features
        self.threshold = 0.5  # Default classification threshold

        # Load hyperparameter search space from JSON file
        with open(param_config, 'r') as file:
            self.param_space = json.load(file)

        self.xgb_model = None  # Placeholder for trained XGBoost model
        self.experiment_name = dbutils.widgets.get('MLFLOW_DIR')  # MLflow experiment name

    def _calculate_metrics(self, y_true, y_prob, threshold=None):
        """
        Calculates model evaluation metrics based on predicted probabilities.
        Now uses actual AUC of precision-recall curve instead of average precision.
        """
        if threshold is None:
            threshold = self.threshold

        # Convert probabilities to binary predictions
        y_pred = (y_prob >= threshold).astype(int)
        
        # Calculate precision-recall curve and compute AUC
        precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
        
        # Calculate AUCPR using trapezoidal rule
        # Reverse order since recalls are in descending order
        aucpr = auc(recalls[::-1], precisions[::-1])

        return {
            'aucpr': aucpr,  # Using actual AUC of PR curve instead of average_precision_score
            'auc': roc_auc_score(y_true, y_prob),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'accuracy': accuracy_score(y_true, y_pred)
        }
    
    def _optimize_threshold(self, y_true, y_prob):
        """
        Optimize the classification threshold based on F2-score while ensuring:
        - Minimum recall of 70% (if possible)
        - Threshold is not less than 0.4
        - Precision-Recall (PR) curve is logged in MLflow

        Parameters:
        - y_true: Ground truth labels (0 or 1)
        - y_prob: Predicted probabilities from the model

        Returns:
        - selected_threshold: Optimal threshold for classification
        """

        # Compute precision, recall, and thresholds from Precision-Recall curve
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

        # Add the maximum threshold (1.0) for completeness
        thresholds = np.append(thresholds, 1.0)

        # ---- Enforce Minimum Threshold Constraint (≥ 0.5) ----
        valid_thresholds = thresholds >= 0.5
        precisions, recalls, thresholds = precisions[valid_thresholds], recalls[valid_thresholds], thresholds[valid_thresholds]

        # ---- Ensure Recall Constraint (≥ 70%) ----
        valid_points = recalls >= 0.70

        if not np.any(valid_points):
            # No threshold meets recall ≥ 70%, so fall back to maximizing F2-score
            print("Warning: No threshold achieves minimum recall of 0.70 with threshold ≥ 0.4")
            f2_scores = (5 * precisions * recalls) / (4 * precisions + recalls)
            best_idx = np.argmax(f2_scores)  # Find the threshold that maximizes F2-score
        else:
            # Filter precision, recall, and threshold values that satisfy recall ≥ 70%
            valid_precisions = precisions[valid_points]
            valid_recalls = recalls[valid_points]
            valid_thresholds = thresholds[valid_points]

            # Compute F2-score (higher weight on recall)
            f2_scores = (5 * valid_precisions * valid_recalls) / (4 * valid_precisions + valid_recalls)
            best_idx = np.argmax(f2_scores)  # Select the best threshold based on F2-score

        # Select the optimal threshold
        selected_threshold = thresholds[best_idx]

        # Compute model performance metrics using the chosen threshold
        metrics = self._calculate_metrics(y_true, y_prob, selected_threshold)

        # Print optimized threshold and key performance metrics
        print(f"\nAt selected threshold {selected_threshold:.4f}:")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"F2 Score: {(5 * metrics['precision'] * metrics['recall']) / (4 * metrics['precision'] + metrics['recall']):.4f}")    

        # ---- Plot and Log Precision-Recall Curve in MLflow ----
        plt.figure(figsize=(8, 6))
        plt.plot(recalls, precisions, marker='.', label="Precision-Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.grid()

        # Save the PR curve plot as an image
        pr_curve_path = "/tmp/pr_curve.png"
        plt.savefig(pr_curve_path)
        plt.close()

        # Log the PR curve image to MLflow under 'plots' directory
        mlflow.log_artifact(pr_curve_path, artifact_path="plots")

        return selected_threshold  # Return the optimized threshold

    def _generate_param_combinations(self):
        """
        Generates all possible hyperparameter combinations from the defined search space.

        Returns:
        - List of dictionaries, each representing a unique hyperparameter combination.
        """
        param_names = list(self.param_space.keys())  # Extract hyperparameter names
        param_values = list(self.param_space.values())  # Extract corresponding value ranges
        
        # Generate all possible combinations using itertools.product
        return [dict(zip(param_names, combo)) for combo in itertools.product(*param_values)]

    def _tune_xgboost(self, X_train, y_train, X_val, y_val):
        """
        Performs hyperparameter tuning using StratifiedKFold cross-validation.
        """
        print("\nTuning XGBoost hyperparameters...")

        with mlflow.start_run(nested=True, run_name="xgboost_tuning", 
                            experiment_id=mlflow.get_experiment_by_name(self.experiment_name).experiment_id):
            
            best_score = -float('inf')
            best_params = None
            
            # Use StratifiedKFold for cross-validation
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            # Iterate over all possible hyperparameter combinations
            for params in self._generate_param_combinations():
                cv_scores = []

                # Perform StratifiedKFold cross-validation
                for train_idx, val_idx in skf.split(X_train, y_train):
                    X_cv_train, X_cv_val = X_train[train_idx], X_train[val_idx]
                    y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]

                    # Initialize XGBoost model with given parameters
                    model = xgb.XGBClassifier(
                        **params,
                        tree_method='hist',
                        eval_metric=['aucpr']
                    )

                    # Train model using early stopping
                    model.fit(
                        X_cv_train, y_cv_train,
                        eval_set=[(X_cv_val, y_cv_val)],
                        early_stopping_rounds=30,
                        verbose=False
                    )

                    # Get predicted probabilities for validation set
                    y_prob = model.predict_proba(X_cv_val)[:, 1]
                    
                    # Calculate precision-recall curve and compute AUC
                    precisions, recalls, _ = precision_recall_curve(y_cv_val, y_prob)
                    # Calculate AUCPR using trapezoidal rule
                    aucpr = auc(recalls[::-1], precisions[::-1])
                    cv_scores.append(aucpr)

                # Compute average cross-validation score
                mean_score = np.mean(cv_scores)

                # If current model performs better, update best parameters
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = params

            # Log best parameters to MLflow
            mlflow.log_params(best_params)

            print("\nBest parameters:")
            for param, value in best_params.items():
                print(f"{param}: {value}")

            # Store the best XGBoost model with optimized parameters
            self.xgb_model = xgb.XGBClassifier(
                **best_params,
                tree_method='hist',
                eval_metric=['aucpr']
            )

    def train(self, X_train, y_train, X_val, y_val):
        """
        Trains the XGBoost model using optimized hyperparameters.

        Parameters:
        - X_train: Training feature matrix
        - y_train: Training labels
        - X_val: Validation feature matrix
        - y_val: Validation labels

        Updates:
        - Trains and stores the final model in self.xgb_model.
        - Optimizes and sets a decision threshold based on validation set.
        """
        print("\nTraining XGBoost model...")
        
        # Tune hyperparameters and select the best model
        self._tune_xgboost(X_train, y_train, X_val, y_val)

        # Train the final model on the full training data
        self.xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=30,
            verbose=False
        )

        # Compute validation probabilities and optimize classification threshold
        y_val_prob = self.xgb_model.predict_proba(X_val)[:, 1]
        self.threshold = self._optimize_threshold(y_val, y_val_prob)

        print(f"\nOptimized threshold: {self.threshold:.4f}")

    def predict_proba(self, X):
        """
        Predicts fraud probabilities using the trained model.

        Parameters:
        - X: Feature matrix

        Returns:
        - Probability of fraud (1) for each sample.
        """
        return self.xgb_model.predict_proba(X)[:, 1]

    def predict(self, X):
        """
        Predicts fraud cases using the optimized threshold.

        Parameters:
        - X: Feature matrix

        Returns:
        - Binary predictions (1 = fraud, 0 = non-fraud).
        """
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def evaluate(self, X, y):
        """
        Evaluates the model on a given dataset.

        Parameters:
        - X: Feature matrix
        - y: True labels

        Returns:
        - Dictionary containing evaluation metrics.
        """
        y_prob = self.predict_proba(X)
        return self._calculate_metrics(y, y_prob)

def train_fraud_detection_system(raw_data):
    """
    Trains the fraud detection system using XGBoost.

    Steps:
    - Splits raw data into training, validation, and test sets.
    - Preprocesses the data (feature engineering, transformations).
    - Initializes and trains an XGBoost model with hyperparameter tuning.
    - Optimizes the decision threshold based on AUCPR.
    - Evaluates model performance and logs metrics in MLflow.
    
    Parameters:
    - raw_data: Pandas DataFrame containing the transaction dataset.
    """

    print("Starting fraud detection system training...")

    # ---- Step 1: Data Splitting ----
    # Split dataset into training, validation, and test sets based on time-based logic.
    train_df, test_df = get_train_test_set(
        raw_data, 
        start_date_training=raw_data['TX_DATETIME'].min(),  # Use earliest available date
        delta_train=7,  # Training period of 7 days
        delta_delay=3,   # Delay period of 3 days (to avoid data leakage)
        delta_test=7     # Testing period of 7 days
    )

    def prep_data(df, training=False):
        df['TX_DURING_WEEKEND'] = df['TX_DATETIME'].apply(is_weekend)
        df['TX_DURING_NIGHT'] = df['TX_DATETIME'].apply(is_night)
        df = df.groupby('CUSTOMER_ID').apply(lambda x: get_customer_spending_features(x, windows_size_in_days=[1, 7, 30]))
        df = df.sort_values('TX_DATETIME').reset_index(drop=True)
        df = df.groupby('TERMINAL_ID').apply(lambda x: get_count_risk_rolling_window(x, delay_period=7, windows_size_in_days=[1, 7, 30], feature="TERMINAL_ID"))
        df = df.sort_values('TX_DATETIME').reset_index(drop=True)
        features = df.drop(['TX_DATETIME', 'TX_FRAUD', 'TX_FRAUD_SCENARIO'], axis=1).columns.to_list()
        X = df.drop(['TX_DATETIME', 'TX_FRAUD', 'TX_FRAUD_SCENARIO'], axis=1).to_numpy()
        y = df['TX_FRAUD'].to_numpy()
        if training is True:
            return X, y, features
        return X, y
    

    # ---- Step 2: Data Preprocessing ----
    print("Preprocessing training data...")
    X_train, y_train, train_features = prep_data(train_df, training=True)

    print("Preprocessing validation data...")
    # Split test set into validation and final test set while maintaining fraud ratio
    grouped = test_df.groupby('TX_FRAUD')
    test_df_final, val_df = [], []

    for _, group in grouped:
        mid = len(group) // 2  # Split each fraud/non-fraud group evenly
        test_df_final.append(group.iloc[:mid])  # First half goes to test set
        val_df.append(group.iloc[mid:])  # Second half goes to validation set

    test_df_final = pd.concat(test_df_final, axis=0).reset_index(drop=True)
    val_df = pd.concat(val_df, axis=0).reset_index(drop=True)
    X_val, y_val = prep_data(val_df)

    print("Preprocessing test data...")
    X_test, y_test = prep_data(test_df_final)

    # ---- Step 3: Log Data Statistics ----
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
    
    # Print dataset statistics
    for stat, value in data_stats.items():
        print(f"{stat}: {value}")

    # ---- Step 4: Initialize XGBoost Model ----
    print("\nInitializing XGBoost model...")
    feature_names = train_features  # Store feature names for reference
    model = FraudDetectionXGBoost(
        input_dim=X_train.shape[1],  # Set input dimensions
        feature_names=feature_names
    )

    # ---- Step 5: Train Model and Log with MLflow ----
    exp = dbutils.widgets.get('MLFLOW_DIR')  # Get MLflow experiment directory
    with mlflow.start_run(run_name="fraud_detection_xgboost", 
                          experiment_id=mlflow.get_experiment_by_name(exp).experiment_id):
        
        # Log dataset statistics for tracking
        mlflow.log_params(data_stats)

        # Train XGBoost model with hyperparameter tuning and threshold optimization
        model.train(X_train, y_train, X_val, y_val)

        # ---- Step 6: Evaluate Model Performance ----
        print("\nEvaluating on test data...")
        test_metrics = model.evaluate(X_test, y_test)

        # Print test set evaluation metrics
        print("\nTest Metrics:")
        for metric, value in test_metrics.items():
            print(f"{metric}: {value:.4f}")

        # Log model evaluation metrics to MLflow
        mlflow.log_metrics(test_metrics)

        # ---- Step 7: Log Trained Model to MLflow ----
        input_example = pd.DataFrame(X_train[:1], columns=feature_names)  # Example input data

        # Log XGBoost model
        base_signature = mlflow.models.signature.infer_signature(
            X_train, 
            model.xgb_model.predict_proba(X_train)  # Ensure probability-based output
        )
        mlflow.sklearn.log_model(
            sk_model=model.xgb_model,
            artifact_path="xgboost_base_model",
            input_example=input_example,
            signature=base_signature,
            registered_model_name="fraud_detection_xgboost_base"
        )

        # Log optimized classification threshold
        mlflow.log_param("optimal_threshold", model.threshold)

        # ---- Step 8: Log Precision-Recall Curve ----
        print("\nLogging Precision-Recall Curve to MLflow...")
        precisions, recalls, _ = precision_recall_curve(y_test, model.predict_proba(X_test))

        plt.figure(figsize=(8, 6))
        plt.plot(recalls, precisions, marker='.', label="Precision-Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.grid()

        # Save and log PR curve image
        pr_curve_path = "/tmp/pr_curve.png"
        plt.savefig(pr_curve_path)
        plt.close()

        mlflow.log_artifact(pr_curve_path, artifact_path="plots")

    print("\nXGBoost training and evaluation completed successfully.")

# ---- Data Loading ----
directory = dbutils.widgets.get('DIR_NAME')
df_list = []

# Load all CSV files in the specified directory
for file in os.listdir(directory):
    if file.endswith('.csv'):
        file_path = os.path.join(directory, file)
        df = pd.read_csv(file_path)
        df_list.append(df)
# Combine all loaded data into a single DataFrame
df = pd.concat(df_list, ignore_index=True)
# Drop unnecessary index column
df = df.drop(df.columns[0], axis=1)

# Train the fraud detection model
train_fraud_detection_system(df) 