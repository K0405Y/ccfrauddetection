import holidays.utils
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from category_encoders import TargetEncoder
import holidays


class TransactionPreprocessor:
    def __init__(self):
        self.robust_scaler = RobustScaler()
        self.standard_scaler = StandardScaler()
        self.target_encoder = TargetEncoder()
        self.us_holidays = holidays.US()
        
    def _extract_datetime_features(self, df):
        """Extract temporal features from TX_DATETIME"""
        df['TX_DATETIME'] = pd.to_datetime(df['TX_DATETIME'])
        
        # Time-based features
        df['hour'] = df['TX_DATETIME'].dt.hour
        df['day_of_week'] = df['TX_DATETIME'].dt.dayofweek
        df['month'] = df['TX_DATETIME'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Time windows for different activities
        df['is_night'] = ((df['hour'] >= 23) | (df['hour'] <= 4)).astype(int)
        df['is_rush_hour'] = (((df['hour'] >= 8) & (df['hour'] <= 10)) | 
                             ((df['hour'] >= 16) & (df['hour'] <= 18))).astype(int)
        
        # Holiday feature
        df['is_holiday'] = df['TX_DATETIME'].apply(
            lambda x: x in self.us_holidays).astype(int)
        
        return df
    
    def _create_amount_features(self, df):
        """Create amount-based features"""
        # Amount transformations
        df['amount_log'] = np.log1p(df['TX_AMOUNT'])
        
        # Round amount features
        df['amount_rounded'] = df['TX_AMOUNT'].round(-1).astype(int)
        df['is_round_amount'] = (df['TX_AMOUNT'] % 10 == 0).astype(int)
        
        # Amount statistics per customer
        customer_amounts = df.groupby('CUSTOMER_ID')['TX_AMOUNT'].agg([
            'mean', 'std', 'max', 'min'
        ]).fillna(0)
        
        df = df.merge(customer_amounts, on='CUSTOMER_ID', how='left')
        
        # Amount deviation from customer mean
        df['amount_deviation'] = abs(df['TX_AMOUNT'] - df['mean']) / df['std'].clip(lower=1)
        
        return df
    
    def _create_customer_features(self, df):
        """Create customer behavior features"""
        # Transaction frequency
        customer_tx_counts = df.groupby('CUSTOMER_ID').size().reset_index(name='customer_tx_count')
        df = df.merge(customer_tx_counts, on='CUSTOMER_ID', how='left')
        
        # Terminal usage patterns
        customer_terminals = df.groupby('CUSTOMER_ID')['TERMINAL_ID'].nunique().reset_index(
            name='customer_terminal_count')
        df = df.merge(customer_terminals, on='CUSTOMER_ID', how='left')
        
        # Time patterns
        customer_hours = df.groupby('CUSTOMER_ID')['hour'].agg(['mean', 'std']).reset_index()
        customer_hours.columns = ['CUSTOMER_ID', 'customer_hour_mean', 'customer_hour_std']
        df = df.merge(customer_hours, on='CUSTOMER_ID', how='left')
        
        return df
    
    def _create_terminal_features(self, df):
        """Create terminal behavior features"""
        # Transaction frequency at terminal
        terminal_tx_counts = df.groupby('TERMINAL_ID').size().reset_index(name='terminal_tx_count')
        df = df.merge(terminal_tx_counts, on='TERMINAL_ID', how='left')
        
        # Amount patterns at terminal
        terminal_amounts = df.groupby('TERMINAL_ID')['TX_AMOUNT'].agg([
            'mean', 'std'
        ]).reset_index()
        terminal_amounts.columns = ['TERMINAL_ID', 'terminal_amount_mean', 'terminal_amount_std']
        df = df.merge(terminal_amounts, on='TERMINAL_ID', how='left')
        
        # Fraud rate at terminal
        terminal_fraud_rate = df.groupby('TERMINAL_ID')['TX_FRAUD'].mean().reset_index(
            name='terminal_fraud_rate')
        df = df.merge(terminal_fraud_rate, on='TERMINAL_ID', how='left')
        
        return df
    
    def _create_sequence_features(self, df):
        """Create sequence-based features"""
        df = df.sort_values(['CUSTOMER_ID', 'TX_DATETIME'])
        
        # Time since last transaction
        df['time_since_last'] = df.groupby('CUSTOMER_ID')['TX_DATETIME'].diff().dt.total_seconds()
        
        # Amount difference from last transaction
        df['amount_diff_last'] = df.groupby('CUSTOMER_ID')['TX_AMOUNT'].diff()
        
        # Terminal changes
        df['terminal_changed'] = (df.groupby('CUSTOMER_ID')['TERMINAL_ID'].shift() != 
                                df['TERMINAL_ID']).astype(int)
        
        return df
    
    def _scale_features(self, df, feature_groups):
        """Scale features by group"""
        scaled_groups = {}
        
        for group, columns in feature_groups.items():
            if group in ['amount', 'sequence']:
                # Use RobustScaler for amount and sequence features
                scaled_groups[group] = self.robust_scaler.fit_transform(df[columns])
            else:
                # Use StandardScaler for other numerical features
                scaled_groups[group] = self.standard_scaler.fit_transform(df[columns])
        
        return scaled_groups
    
    def transform(self, df, training=True):
        """Main transformation pipeline"""
        # Create copy to avoid modifying original data
        df = df.copy()
        
        # Extract datetime features
        df = self._extract_datetime_features(df)
        
        # Create feature groups
        df = self._create_amount_features(df)
        df = self._create_customer_features(df)
        df = self._create_terminal_features(df)
        df = self._create_sequence_features(df)
        
        # Define feature groups
        feature_groups = {
            'customer': [
                'customer_tx_count', 'customer_terminal_count',
                'customer_hour_mean', 'customer_hour_std'
            ],
            'terminal': [
                'terminal_tx_count', 'terminal_amount_mean',
                'terminal_amount_std', 'terminal_fraud_rate'
            ],
            'amount': [
                'TX_AMOUNT', 'amount_log', 'amount_deviation',
                'mean', 'std', 'max', 'min'
            ],
            'temporal': [
                'hour', 'day_of_week', 'month', 'is_weekend',
                'is_night', 'is_rush_hour', 'is_holiday'
            ],
            'sequence': [
                'time_since_last', 'amount_diff_last', 'terminal_changed'
            ]
        }
        
        # Scale features
        scaled_features = self._scale_features(df, feature_groups)
        
        if training:
            # Apply SMOTE only during training
            smote = SMOTE(random_state=42)
            
            # Combine all scaled features
            X = np.concatenate([scaled_features[group] for group in feature_groups.keys()], axis=1)
            y = df['TX_FRAUD'].values
            
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            # Split back into feature groups
            start_idx = 0
            resampled_groups = {}
            
            for group, columns in feature_groups.items():
                end_idx = start_idx + len(columns)
                resampled_groups[group] = X_resampled[:, start_idx:end_idx]
                start_idx = end_idx
            
            return resampled_groups, y_resampled
        
        return scaled_features

