import os
import holidays.utils
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from category_encoders import TargetEncoder
import holidays


class TransactionPreprocessor:
    def __init__(self):
        self.scalers = {}
        self.imputers = {}
        self.us_holidays = holidays.US()
        
        # Define feature groups with consistent ordering
        self.feature_groups = {
            'temporal': [
                'hour', 'day_of_week', 'month', 'is_weekend',
                'is_night', 'is_rush_hour', 'is_holiday'
            ],
            'amount': [
                'TX_AMOUNT', 'amount_log', 'amount_deviation',
                'mean', 'std', 'max', 'min'
            ],
            'customer': [
                'customer_tx_count', 'customer_terminal_count',
                'customer_hour_mean', 'customer_hour_std'
            ],
            'terminals': [
                'terminal_tx_count', 'terminal_amount_mean', 'terminal_amount_std',
                'terminal_amount_median', 'terminal_tx_count_large',
                'terminal_tx_time_mean', 'terminal_tx_time_std',
                'terminal_fraud_rate_smoothed'
            ],
            'sequence': [
                'time_since_last', 'time_until_next', 
                'amount_diff_last', 'amount_diff_next',
                'terminal_changed', 'tx_velocity_1h', 'tx_velocity_24h',
                'amount_velocity_1h', 'amount_velocity_24h',
                'repeated_terminal', 'unique_terminals_24h'
            ]
        }
        
        # Initialize scalers and imputers for each group
        for group in self.feature_groups:
            if group in ['amount', 'sequence']:
                self.scalers[group] = RobustScaler()
            else:
                self.scalers[group] = StandardScaler()
            self.imputers[group] = SimpleImputer(strategy='median')
    
    def _extract_datetime_features(self, df):
        """Extract temporal features from TX_DATETIME"""
        df = df.copy()
        try:
            df['TX_DATETIME'] = pd.to_datetime(df['TX_DATETIME'])
        except (ValueError, TypeError):
            df['TX_DATETIME'] = pd.to_datetime(df['TX_DATETIME'], errors='coerce')
        
        # Time-based features
        df['hour'] = df['TX_DATETIME'].dt.hour
        df['day_of_week'] = df['TX_DATETIME'].dt.dayofweek
        df['month'] = df['TX_DATETIME'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_night'] = ((df['hour'] >= 23) | (df['hour'] <= 4)).astype(int)
        df['is_rush_hour'] = (((df['hour'] >= 8) & (df['hour'] <= 10)) | 
                             ((df['hour'] >= 16) & (df['hour'] <= 18))).astype(int)
        df['is_holiday'] = df['TX_DATETIME'].apply(
            lambda x: x in self.us_holidays if pd.notnull(x) else False).astype(int)
        
        return df
    
    def _create_amount_features(self, df):
        """Create amount-based features"""
        df = df.copy()
        df['TX_AMOUNT'] = df['TX_AMOUNT'].clip(lower=0)
        df['amount_log'] = np.log1p(df['TX_AMOUNT'])
        
        # Customer amount statistics
        amount_stats = df.groupby('CUSTOMER_ID')['TX_AMOUNT'].agg(['mean', 'std', 'max', 'min']).fillna(0)
        df = df.merge(amount_stats, on='CUSTOMER_ID', how='left')
        
        # Handle std=0 cases for amount_deviation
        df['std'] = df['std'].replace(0, df['std'].mean())
        df['amount_deviation'] = abs(df['TX_AMOUNT'] - df['mean']) / df['std']
        
        return df
    
    def _create_customer_features(self, df):
        """Create customer-related features"""
        df = df.copy()
        
        # Transaction counts
        customer_stats = df.groupby('CUSTOMER_ID').agg({
            'TX_DATETIME': 'count',
            'TERMINAL_ID': 'nunique',
            'hour': ['mean', 'std']
        }).fillna(0)
        
        customer_stats.columns = ['customer_tx_count', 'customer_terminal_count', 
                                'customer_hour_mean', 'customer_hour_std']
        
        df = df.merge(customer_stats, on='CUSTOMER_ID', how='left')
        return df
    
    def _create_terminal_features(self, df):
        """Create terminal-related features"""
        df = df.copy()
        
        # Terminal transaction statistics
        terminal_stats = df.groupby('TERMINAL_ID').agg({
            'TX_AMOUNT': ['count', 'mean', 'std', 'median'],
            'TX_DATETIME': lambda x: x.diff().mean().total_seconds() if len(x) > 1 else 0
        }).fillna(0)
        
        terminal_stats.columns = ['terminal_tx_count', 'terminal_amount_mean', 
                                'terminal_amount_std', 'terminal_amount_median',
                                'terminal_tx_time_mean']
        
        # Additional terminal features
        terminal_stats['terminal_tx_count_large'] = df.groupby('TERMINAL_ID')['TX_AMOUNT'].apply(
            lambda x: (x > x.mean()).sum()).fillna(0)
        terminal_stats['terminal_tx_time_std'] = df.groupby('TERMINAL_ID')['TX_DATETIME'].apply(
            lambda x: x.diff().std().total_seconds() if len(x) > 1 else 0).fillna(0)
        
        # Terminal fraud rate
        terminal_fraud = df.groupby('TERMINAL_ID')['TX_FRAUD'].agg(['mean', 'sum']).fillna(0)
        terminal_stats['terminal_fraud_rate_smoothed'] = (terminal_fraud['sum'] + 0.01) / (terminal_stats['terminal_tx_count'] + 0.02)
        
        df = df.merge(terminal_stats, on='TERMINAL_ID', how='left')
        return df
    
    def _create_sequence_features(self, df):
        """Create sequence-related features"""
        df = df.copy()
        df = df.sort_values(['CUSTOMER_ID', 'TX_DATETIME'])
        
        # Time-based sequence features
        df['time_since_last'] = df.groupby('CUSTOMER_ID')['TX_DATETIME'].diff().dt.total_seconds().fillna(0)
        df['time_until_next'] = df.groupby('CUSTOMER_ID')['TX_DATETIME'].diff(-1).dt.total_seconds().fillna(0)
        
        # Amount sequence features
        df['amount_diff_last'] = df.groupby('CUSTOMER_ID')['TX_AMOUNT'].diff().fillna(0)
        df['amount_diff_next'] = df.groupby('CUSTOMER_ID')['TX_AMOUNT'].diff(-1).fillna(0)
        
        # Terminal sequence features
        df['terminal_changed'] = (df.groupby('CUSTOMER_ID')['TERMINAL_ID'].shift() != df['TERMINAL_ID']).astype(int).fillna(1)
        df['repeated_terminal'] = df.groupby(['CUSTOMER_ID', 'TERMINAL_ID']).cumcount()
        
        # Velocity features
        for window in ['1h', '24h']:
            td = pd.Timedelta(window)
            for customer_id in df['CUSTOMER_ID'].unique():
                mask = df['CUSTOMER_ID'] == customer_id
                customer_data = df[mask].copy()
                
                # Calculate rolling counts and amounts
                for idx in customer_data.index:
                    time = customer_data.loc[idx, 'TX_DATETIME']
                    window_mask = (customer_data['TX_DATETIME'] <= time) & (customer_data['TX_DATETIME'] > time - td)
                    
                    df.loc[idx, f'tx_velocity_{window}'] = window_mask.sum()
                    df.loc[idx, f'amount_velocity_{window}'] = customer_data.loc[window_mask, 'TX_AMOUNT'].sum()
                    if window == '24h':
                        df.loc[idx, 'unique_terminals_24h'] = customer_data.loc[window_mask, 'TERMINAL_ID'].nunique()
        
        return df
    
    def _create_features(self, df):
        """Create all features in consistent order"""
        df = df.copy()
        
        # Create features
        df = self._extract_datetime_features(df)
        df = self._create_amount_features(df)
        df = self._create_customer_features(df)
        df = self._create_terminal_features(df)
        df = self._create_sequence_features(df)
        
        # Ensure all features exist with correct ordering
        for group, features in self.feature_groups.items():
            missing_features = set(features) - set(df.columns)
            for feature in missing_features:
                df[feature] = 0
        
        return df
    
    def transform(self, df, training=True):
        """Transform data with consistent feature handling"""
        print("Starting feature transformation...")
        df = df.copy()
        
        # Create all features
        df = self._create_features(df)
        
        # Process each feature group
        transformed_groups = {}
        for group, features in self.feature_groups.items():
            # Extract and order features
            group_data = df[features].copy()
            
            # Impute missing values
            if training:
                group_data = pd.DataFrame(
                    self.imputers[group].fit_transform(group_data),
                    columns=features
                )
            else:
                group_data = pd.DataFrame(
                    self.imputers[group].transform(group_data),
                    columns=features
                )
            
            # Scale features
            if training:
                transformed_groups[group] = self.scalers[group].fit_transform(group_data)
            else:
                transformed_groups[group] = self.scalers[group].transform(group_data)
        
        if training:
            # Combine features for SMOTE
            X = np.concatenate([transformed_groups[group] for group in self.feature_groups.keys()], axis=1)
            y = df['TX_FRAUD'].values
            
            # Apply SMOTE
            n_minority = (y == 1).sum()
            n_majority = (y == 0).sum()
            target_minority = int(n_majority * 0.5)
            
            print(f"\nTraining Data Class distribution before SMOTE:")
            print(f"Non-fraud: {n_majority}, Fraud: {n_minority}")
            
            smote = SMOTE(
                sampling_strategy={1: target_minority},
                random_state=42,
                k_neighbors=min(5, n_minority - 1)
            )
            
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            print(f"\nTraining Data Class distribution after SMOTE:")
            print(f"Non-fraud: {sum(y_resampled == 0)}, Fraud: {sum(y_resampled == 1)}")
            
            # Split back into feature groups
            start_idx = 0
            resampled_groups = {}
            for group in self.feature_groups.keys():
                end_idx = start_idx + len(self.feature_groups[group])
                resampled_groups[group] = X_resampled[:, start_idx:end_idx]
                start_idx = end_idx
            
            return resampled_groups, y_resampled
        
        return transformed_groups