import os
import sys
from src.preprocessing.prep import TransactionPreprocessor
import mlflow
import pandas as pd
import numpy as np
import pickle as pkl
from mlflow.pyfunc import PythonModel
from datetime import datetime, timedelta
import torch
from typing import Dict, Any, Union

class FraudDetectionEnsemble(PythonModel):
    def __init__(self, model_versions, data_directory):
        """
        Initialize with model versions and directory containing transaction CSV files
        
        Args:
            model_versions (dict): Model version mapping
            data_directory (str): Path to directory containing transaction CSV files
        """
        self.model_versions = model_versions
        self.data_directory = data_directory
        self.xgb_model = None
        self.rf_model = None
        self.nn_model = None
        self.feature_names = None
        self.weights = [0.4, 0.3, 0.3]
        
    def _load_customer_transactions(self, customer_id: int) -> pd.DataFrame:
        """
        load historical transactions for a specific customer
        """
        customer_data = []
        customer_found = True
        required_columns = ['CUSTOMER_ID', 'TERMINAL_ID', 'TX_AMOUNT', 'TX_DATETIME', 
                          'TX_TIME_SECONDS', 'TX_TIME_DAYS']
          
        # Iterate through txn files in directory
        for filename in os.listdir(self.data_directory):
            if filename.endswith('.csv'):
                file_path = os.path.join(self.data_directory, filename)
                # Read only required columns and filter for customer_id
                try:
                    # First, check if customer_id exists in this file
                    customer_check = pd.read_csv(
                        file_path, 
                        usecols=['CUSTOMER_ID'], 
                        dtype={'CUSTOMER_ID': int}
                    )
    
                    if customer_id in customer_check['CUSTOMER_ID'].values:
                        # Read only required columns with appropriate dtypes
                        df = pd.read_csv(
                            file_path,
                            usecols=required_columns,
                            dtype={
                                'CUSTOMER_ID': int,
                                'TERMINAL_ID': int,
                                'TX_AMOUNT': float,
                                'TX_TIME_SECONDS': float,
                                'TX_TIME_DAYS': float
                            },
                            parse_dates=['TX_DATETIME']
                        )
                        
                        df['TX_DATETIME'] = pd.to_datetime(df['TX_DATETIME'], errors='coerce')
                        df = df.dropna(subset=['TX_DATETIME'])

                        # Filter for specific customer
                        customer_df = df[df['CUSTOMER_ID'] == customer_id]
                        
                        if not customer_df.empty:
                            customer_data.append(customer_df)
                
                except Exception as e:
                    print(f"Error reading file {filename}: {str(e)}")
                    continue

        if not customer_found:
            raise ValueError(f"Customer ID {customer_id} not found in transaction history")

        # if not customer_data:
        #     # Return empty DataFrame with correct columns if no data found
        #     return pd.DataFrame(columns=required_columns)
        
        # Combine all customer data and sort by datetime
        customer_history = pd.concat(customer_data, ignore_index=True)
        return customer_history.sort_values('TX_DATETIME')    
    

    def _calculate_customer_amount_features(self, customer_txns: pd.DataFrame, 
                                         current_amount: float,
                                         tx_datetime: pd.Timestamp) -> Dict[str, float]:
        """Calculate amount features based on customer's transaction history"""

        # Filter for transactions before current timestamp
        past_txns = customer_txns[customer_txns['TX_DATETIME'] < tx_datetime]
        
        if past_txns.empty:
            return {
                'amount': current_amount,
                'amount_log': np.log1p(current_amount),
                'amount_rounded': round(current_amount, -1),
                'is_round_amount': 1 if current_amount % 10 == 0 else 0,
                'amount_mean': current_amount,
                'amount_std': 0.1 * current_amount,
                'amount_max': current_amount,
                'amount_min': current_amount,
                'amount_deviation': 0.0
            }
        
        # Calculate amount statistics
        amount_stats = past_txns['TX_AMOUNT'].agg(['mean', 'std', 'max', 'min'])
        
        # Handle std=0 case
        if pd.isna(amount_stats['std']) or amount_stats['std'] == 0:
            amount_stats['std'] = 0.1 * amount_stats['mean']
            
        amount_deviation = float(abs(current_amount - amount_stats['mean']) / amount_stats['std'])
        
        return {
            'amount': float(current_amount),
            'amount_log': float(np.log1p(current_amount)),
            'amount_rounded': float(round(current_amount, -1)),
            'is_round_amount':float(1 if current_amount % 10 == 0 else 0),
            'amount_mean': float(amount_stats['mean']),
            'amount_std': float(amount_stats['std']),
            'amount_max': float(amount_stats['max']),
            'amount_min': float(amount_stats['min']),
            'amount_deviation': amount_deviation
        }

    def _calculate_sequence_features(self, customer_txns: pd.DataFrame,
                                  terminal_id: int, amount: float,
                                  tx_datetime: pd.Timestamp) -> Dict[str, float]:
        """Calculate sequence features from customer transaction history"""

        customer_txns['TX_DATETIME'] = pd.to_datetime(customer_txns['TX_DATETIME'], errors='coerce')

        past_txns = customer_txns[customer_txns['TX_DATETIME'] < tx_datetime]
        
        if past_txns.empty:
            return {
                'time_since_last': 86400,
                'time_until_next': 86400,
                'amount_diff_last': 0,
                'amount_diff_next': 0,
                'terminal_changed': 1,
                'tx_velocity_1h': 0,
                'tx_velocity_24h': 0,
                'amount_velocity_1h': 0,
                'amount_velocity_24h': 0,
                'unique_terminals_24h': 0,
                'repeated_terminal': 0
            }
        
        # Time windows
        one_hour_ago = tx_datetime - timedelta(hours=1)
        one_day_ago = tx_datetime - timedelta(days=1)
        
        # Get last transaction
        last_tx = past_txns.iloc[-1]
        
        # Calculate time-based features
        time_since_last = (tx_datetime - last_tx['TX_DATETIME']).total_seconds()
        amount_diff = amount - last_tx['TX_AMOUNT']
        terminal_changed = 1 if last_tx['TERMINAL_ID'] != terminal_id else 0
        
        # Calculate velocity features using vectorized operations
        mask_1h = (past_txns['TX_DATETIME'] > one_hour_ago)
        mask_24h = (past_txns['TX_DATETIME'] > one_day_ago)
        
        txns_1h = past_txns[mask_1h]
        txns_24h = past_txns[mask_24h]
        
        return {
            'time_since_last': time_since_last,
            'time_until_next': 0,
            'amount_diff_last': amount_diff,
            'amount_diff_next': 0,
            'terminal_changed': terminal_changed,
            'tx_velocity_1h': len(txns_1h),
            'tx_velocity_24h': len(txns_24h),
            'amount_velocity_1h': txns_1h['TX_AMOUNT'].sum(),
            'amount_velocity_24h': txns_24h['TX_AMOUNT'].sum(),
            'unique_terminals_24h': txns_24h['TERMINAL_ID'].nunique(),
            'repeated_terminal': past_txns[past_txns['TERMINAL_ID'] == terminal_id].shape[0]
        }

    def _calculate_terminal_features(self, terminal_id: int, tx_datetime: pd.Timestamp) -> Dict[str, float]:
        """
        Calculate terminal behavior features from historical data
        """
        # Load terminal transactions
        terminal_txns = []
        required_columns = ['TERMINAL_ID', 'TX_AMOUNT', 'TX_DATETIME', 'TX_FRAUD']
        
        for filename in os.listdir(self.data_directory):
            if filename.endswith('.csv'):
                file_path = os.path.join(self.data_directory, filename)
                
                # First check if terminal exists in file
                terminal_check = pd.read_csv(
                    file_path,
                    usecols=['TERMINAL_ID'],
                    dtype={'TERMINAL_ID': int}
                )
                
                if terminal_id in terminal_check['TERMINAL_ID'].values:
                    df = pd.read_csv(
                        file_path,
                        usecols=required_columns,
                        dtype={
                            'TERMINAL_ID': int,
                            'TX_AMOUNT': float,
                            'TX_FRAUD': int
                        },
                        parse_dates=['TX_DATETIME']
                    )
                    
                    terminal_df = df[df['TERMINAL_ID'] == terminal_id]
                    if not terminal_df.empty:
                        terminal_txns.append(terminal_df)
        
        if not terminal_txns:
            # Return default values if no history found
            return {
                'terminal_tx_count': 1,
                'terminal_amount_mean': 0,
                'terminal_amount_std': 0,
                'terminal_amount_median': 0,
                'terminal_tx_count_large': 0,
                'terminal_tx_time_mean': 86400,  # 1 day in seconds
                'terminal_tx_time_std': 3600,    # 1 hour in seconds
                'terminal_fraud_rate_smoothed': 0.01
            }
        
        # Combine all terminal transactions and filter for past transactions
        terminal_history = pd.concat(terminal_txns, ignore_index=True)
        past_txns = terminal_history[terminal_history['TX_DATETIME'] < tx_datetime]
        
        if past_txns.empty:
            return {
                'terminal_tx_count': 1,
                'terminal_amount_mean': 0,
                'terminal_amount_std': 0,
                'terminal_amount_median': 0,
                'terminal_tx_count_large': 0,
                'terminal_tx_time_mean': 86400,
                'terminal_tx_time_std': 3600,
                'terminal_fraud_rate_smoothed': 0.01
            }
        
        # Calculate amount statistics
        amount_stats = past_txns['TX_AMOUNT'].agg(['mean', 'std', 'median'])
        
        # Calculate large transaction count (transactions above mean)
        large_tx_count = (past_txns['TX_AMOUNT'] > amount_stats['mean']).sum()
        
        # Calculate time between transactions
        past_txns_sorted = past_txns.sort_values('TX_DATETIME')
        time_diffs = past_txns_sorted['TX_DATETIME'].diff().dt.total_seconds()
        time_mean = time_diffs.mean() if len(time_diffs) > 1 else 86400
        time_std = time_diffs.std() if len(time_diffs) > 1 else 3600
        
        # Calculate fraud rate with Laplace smoothing
        total_txns = len(past_txns)
        fraud_count = past_txns['TX_FRAUD'].sum()
        alpha = 0.01  # smoothing parameter
        fraud_rate_smoothed = (fraud_count + alpha) / (total_txns + 2 * alpha)
        
        return {
            'terminal_tx_count': total_txns,
            'terminal_amount_mean': amount_stats['mean'],
            'terminal_amount_std': amount_stats['std'] if pd.notnull(amount_stats['std']) else 0,
            'terminal_amount_median': amount_stats['median'],
            'terminal_tx_count_large': large_tx_count,
            'terminal_tx_time_mean': time_mean,
            'terminal_tx_time_std': time_std,
            'terminal_fraud_rate_smoothed': fraud_rate_smoothed
        }

    def _preprocess_input(self, data):
        """Preprocess input with efficient customer-level feature calculation"""
        if isinstance(data, dict):
            if 'inputs' in data:
                data = data['inputs']
            if isinstance(data, dict):
                data = pd.DataFrame([data])
        
        # Extract base values
        customer_id = int(data['CUSTOMER_ID'].iloc[0])
        terminal_id = int(data['TERMINAL_ID'].iloc[0])
        amount = float(data['TX_AMOUNT'].iloc[0])
        tx_datetime = pd.to_datetime(data['TX_DATETIME'].iloc[0])
        
        # Load customer's transaction history
        customer_txns = self._load_customer_transactions(customer_id)
        
        # Calculate features using the loaded history
        amount_features = self._calculate_customer_amount_features(
            customer_txns, amount, tx_datetime
        )
        
        sequence_features = self._calculate_sequence_features(
            customer_txns, terminal_id, amount, tx_datetime
        )
        
        terminal_features = self._calculate_terminal_features(
            terminal_id, tx_datetime
        )

        # Create features DataFrame
        features = pd.DataFrame(index=[0])
        
        # Map all features
        features['feature_1'] = customer_id / 10000
        features['feature_2'] = float(data['TX_TIME_SECONDS'].iloc[0]) / 86400
        features['feature_3'] = float(data['TX_TIME_DAYS'].iloc[0]) / 7
        features['feature_4'] = terminal_id / 1000
        features['feature_5'] = amount_features['amount']
        features['feature_6'] = amount_features['amount_log']
        
        # Temporal features
        features['feature_7'] = tx_datetime.hour / 24
        features['feature_8'] = tx_datetime.dayofweek / 7
        features['feature_9'] = tx_datetime.month / 12
        features['feature_10'] = 1 if tx_datetime.dayofweek >= 5 else 0
        features['feature_11'] = 1 if (tx_datetime.hour >= 23 or tx_datetime.hour <= 4) else 0
        features['feature_12'] = 1 if (8 <= tx_datetime.hour <= 10 or 
                                     16 <= tx_datetime.hour <= 18) else 0
        
        # Amount features
        features['feature_13'] = amount_features['amount_deviation']
        features['feature_14'] = amount_features['amount_mean']
        features['feature_15'] = amount_features['amount_std']
        features['feature_16'] = amount_features['amount_max']
        features['feature_17'] = amount_features['amount_min']
        
        # Customer behavior features
        features['feature_18'] = len(customer_txns)
        features['feature_19'] = customer_txns['TERMINAL_ID'].nunique()
        features['feature_20'] = customer_txns['TX_DATETIME'].dt.hour.mean()
        features['feature_21'] = customer_txns['TX_DATETIME'].dt.hour.std()
        
        # Terminal features
        features['feature_22'] = terminal_features['terminal_tx_count']
        features['feature_23'] = terminal_features['terminal_amount_mean']
        features['feature_24'] = terminal_features['terminal_amount_std']
        features['feature_25'] = terminal_features['terminal_amount_median']
        features['feature_26'] = terminal_features['terminal_tx_count_large']
        features['feature_27'] = terminal_features['terminal_tx_time_mean']
        features['feature_28'] = terminal_features['terminal_tx_time_std']
        features['feature_29'] = terminal_features['terminal_fraud_rate_smoothed']
        
        # Sequence features
        features['feature_30'] = sequence_features['time_since_last']
        features['feature_31'] = sequence_features['time_until_next']
        features['feature_32'] = sequence_features['amount_diff_last']
        features['feature_33'] = sequence_features['amount_diff_next']
        features['feature_34'] = sequence_features['terminal_changed']
        features['feature_35'] = sequence_features['tx_velocity_1h']
        features['feature_36'] = sequence_features['tx_velocity_24h']
        features['feature_37'] = sequence_features['repeated_terminal']
        
        return features

    def _load_model(self, workspace, model_name, version):
        return mlflow.pyfunc.load_model(
            model_uri=f"models:/{model_name}/{version}"
        )

    def load_context(self, context):
        """Load models"""
        if "DATABRICKS_RUNTIME_VERSION" in os.environ:
            mlflow.set_tracking_uri("databricks")
        else:
            mlflow.set_tracking_uri("local")
            
        self.xgb_model = self._load_model(context, 'xgb_model', self.model_versions['xgb_model'])
        self.rf_model = self._load_model(context, 'rf_model', self.model_versions['rf_model'])
        self.nn_model = self._load_model(context, 'pytorch_model', self.model_versions['pytorch_model'])
        
        self.feature_names = [f'feature_{i}' for i in range(1, 38)]

    def predict(self, context, input_data):
        """Make predictions using the ensemble"""
        try:
            # Preprocess input data
            X = self._preprocess_input(input_data)
            
            # Get individual model predictions
            xgb_prob = self.xgb_model.predict(X)
            rf_prob = self.rf_model.predict(X)

            X_tensor = torch.tensor(X.values, dtype=torch.float32)
            nn_prob = self.nn_model.predict(pd.DataFrame(X_tensor.numpy()))
            
            # Convert predictions to probabilities if needed
            if hasattr(xgb_prob, 'iloc'):
                xgb_prob = float(xgb_prob.iloc[0])
            elif isinstance(xgb_prob, np.ndarray):
                xgb_prob = float(xgb_prob[0])
            
            if hasattr(rf_prob, 'iloc'):
                rf_prob = float(rf_prob.iloc[0])
            elif isinstance(rf_prob, np.ndarray):
                rf_prob = float(rf_prob[0])
            
            if hasattr(nn_prob, 'iloc'):
                nn_prob = float(nn_prob.iloc[0])
            elif isinstance(nn_prob, np.ndarray):
                nn_prob = float(nn_prob[0])
            
            # Calculate ensemble probability
            ensemble_prob = (
                self.weights[0] * xgb_prob +
                self.weights[1] * rf_prob +
                self.weights[2] * nn_prob
            )
            
            # Make prediction
            prediction = "TRANSACTION IS FRAUDULENT" if ensemble_prob >= 0.5 else "TRANSACTION IS NOT FRAUDULENT"
            
            return [{
                'prediction': prediction,
                'probability': ensemble_prob,
                'model_predictions': {
                    'xgboost': xgb_prob,
                    'random_forest': rf_prob,
                    'neural_network': nn_prob
                }
            }]
            
        except ValueError as e:
            if "Customer ID" in str(e):
                return[{"error": "Invalid customer ID"}]
            raise
        except Exception as e:
            raise RuntimeError(f"Prediction error: {str(e)}")


# Example input
example_input = {
    "TRANSACTION_ID": 4781,
    "TX_DATETIME": "2024-10-29 05:57:40",
    "CUSTOMER_ID": 568,
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
model = FraudDetectionEnsemble(model_versions = model_versions, data_directory='/Workspace/Users/kehinde.awomuti@pwc.com/ccfrauddetection/data')

experiment_id = mlflow.set_experiment('/Users/kehinde.awomuti@pwc.com/fraud_detection_inference')
# Register the model
with mlflow.start_run(run_name='inference_engine') as run:
    mlflow.pyfunc.log_model(
        artifact_path="fraud_detection_inference",
        python_model=model,
        registered_model_name="fraud_detection_model",
        input_example=example_input
    )