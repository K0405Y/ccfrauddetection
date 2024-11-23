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
        self.robust_scaler = RobustScaler()
        self.standard_scaler = StandardScaler()
        self.target_encoder = TargetEncoder()
        self.us_holidays = holidays.US()
        self.numerical_imputer = SimpleImputer(strategy='median')
        self.categorical_imputer = SimpleImputer(strategy='constant', fill_value='missing')
        self.feature_groups = {
            'customer': [
                'customer_tx_count', 'customer_terminal_count',
                'customer_hour_mean', 'customer_hour_std'
            ],
            'terminal': [
                'terminal_tx_count', 'terminal_amount_mean', 'terminal_amount_std',
                'terminal_amount_median', 'terminal_tx_count_large',
                'terminal_tx_time_mean', 'terminal_tx_time_std',
                'terminal_fraud_rate_smoothed'
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
                'time_since_last', 'time_until_next', 
                'amount_diff_last', 'amount_diff_next',
                'terminal_changed', 'tx_velocity_1h', 'tx_velocity_24h',
                'amount_velocity_1h', 'amount_velocity_24h',
                'repeated_terminal', 'unique_terminals_24h'
            ]
        }
        
    def _clean_data(self, df):
        """Clean and handle missing values in the dataset"""
        df = df.copy()
        
        # Convert data types
        df['CUSTOMER_ID'] = pd.to_numeric(df['CUSTOMER_ID'], errors='coerce')
        df['TERMINAL_ID'] = pd.to_numeric(df['TERMINAL_ID'], errors='coerce')
        df['TX_AMOUNT'] = pd.to_numeric(df['TX_AMOUNT'], errors='coerce')
        
        # Handle infinite values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], np.nan)
        
        # Log missing values
        missing_values = df.isnull().sum()
        print("Missing values before imputation:")
        print(missing_values[missing_values > 0])
        
        return df
    
    def _impute_missing_values(self, df, feature_groups):
        """Impute missing values for each feature group"""
        imputed_groups = {}
        
        for group, columns in feature_groups.items():
            group_data = df[columns].copy()
            
            # Check if group contains numeric data
            is_numeric = np.issubdtype(group_data.dtypes.iloc[0], np.number)
            
            if is_numeric:
                imputer = self.numerical_imputer
            else:
                imputer = self.categorical_imputer
            
            # Fit and transform the imputer
            imputed_data = imputer.fit_transform(group_data)
            
            # Convert back to DataFrame
            imputed_groups[group] = pd.DataFrame(
                imputed_data, 
                columns=columns,
                index=df.index
            )
        
        return imputed_groups
    
    def _extract_datetime_features(self, df):
        """Extract temporal features from TX_DATETIME"""
        # Handle potential datetime parsing errors
        try:
            df['TX_DATETIME'] = pd.to_datetime(df['TX_DATETIME'])
        except (ValueError, TypeError):
            print("Warning: Some datetime values could not be parsed")
            df['TX_DATETIME'] = pd.to_datetime(df['TX_DATETIME'], errors='coerce')
        
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
            lambda x: x in self.us_holidays if pd.notnull(x) else False).astype(int)
        
        return df
    
    def _create_amount_features(self, df):
        """Create amount-based features with null handling"""
        # Handle negative amounts
        df['TX_AMOUNT'] = df['TX_AMOUNT'].clip(lower=0)
        
        # Amount transformations
        df['amount_log'] = np.log1p(df['TX_AMOUNT'])
        
        # Round amount features
        df['amount_rounded'] = df['TX_AMOUNT'].round(-1).fillna(-1).astype(int)
        df['is_round_amount'] = (df['TX_AMOUNT'] % 10 == 0).astype(int)
        
        # Amount statistics per customer with null handling
        customer_amounts = df.groupby('CUSTOMER_ID')['TX_AMOUNT'].agg([
            'mean', 'std', 'max', 'min'
        ]).fillna(method='ffill').fillna(0)
        
        df = df.merge(customer_amounts, on='CUSTOMER_ID', how='left')
        
        # Amount deviation from customer mean
        df['std'] = df['std'].replace(0, df['std'].mean())
        df['amount_deviation'] = abs(df['TX_AMOUNT'] - df['mean']) / df['std']
        
        return df
    
    def _create_customer_features(self, df):
        """Create customer behavior features with null handling"""
        # Transaction frequency
        customer_tx_counts = df.groupby('CUSTOMER_ID').size().reset_index(name='customer_tx_count')
        df = df.merge(customer_tx_counts, on='CUSTOMER_ID', how='left')
        
        # Terminal usage patterns
        customer_terminals = df.groupby('CUSTOMER_ID')['TERMINAL_ID'].nunique().reset_index(
            name='customer_terminal_count')
        df = df.merge(customer_terminals, on='CUSTOMER_ID', how='left')
        
        # Time patterns with null handling
        df['hour'] = df['hour'].fillna(df['hour'].median())
        customer_hours = df.groupby('CUSTOMER_ID')['hour'].agg(['mean', 'std']).reset_index()
        customer_hours.columns = ['CUSTOMER_ID', 'customer_hour_mean', 'customer_hour_std']
        customer_hours = customer_hours.fillna(method='ffill').fillna(method='bfill')
        df = df.merge(customer_hours, on='CUSTOMER_ID', how='left')
        
        return df
    
    def _create_terminal_features(self, df):
        """Create terminal behavior features with null handling"""
        # Transaction frequency at terminal
        terminal_tx_counts = df.groupby('TERMINAL_ID').size().reset_index(name='terminal_tx_count')
        df = df.merge(terminal_tx_counts, on='TERMINAL_ID', how='left')
        df['terminal_tx_count'] = df['terminal_tx_count'].fillna(0)
        
        # Amount patterns at terminal with null handling
        terminal_amounts = df.groupby('TERMINAL_ID').agg({
            'TX_AMOUNT': [
                ('terminal_amount_mean', 'mean'),
                ('terminal_amount_std', 'std'),
                ('terminal_amount_median', 'median'),
                ('terminal_tx_count_large', lambda x: (x > x.mean()).sum())
            ]
        }).reset_index()
        
        # Flatten column names
        terminal_amounts.columns = ['TERMINAL_ID', 'terminal_amount_mean', 
                                  'terminal_amount_std', 'terminal_amount_median',
                                  'terminal_tx_count_large']
        
        # Fill NaN values in terminal statistics
        terminal_amounts = terminal_amounts.fillna({
            'terminal_amount_mean': df['TX_AMOUNT'].mean(),
            'terminal_amount_std': df['TX_AMOUNT'].std(),
            'terminal_amount_median': df['TX_AMOUNT'].median(),
            'terminal_tx_count_large': 0
        })
        
        df = df.merge(terminal_amounts, on='TERMINAL_ID', how='left')
        
        # Terminal velocity features
        terminal_velocity = df.groupby('TERMINAL_ID').agg({
            'TX_DATETIME': [
                ('terminal_tx_time_mean', lambda x: x.diff().mean().total_seconds()),
                ('terminal_tx_time_std', lambda x: x.diff().std().total_seconds() if len(x) > 1 else 0)
            ]
        }).reset_index()
        
        # Flatten column names
        terminal_velocity.columns = ['TERMINAL_ID', 'terminal_tx_time_mean', 'terminal_tx_time_std']
        terminal_velocity = terminal_velocity.fillna({
            'terminal_tx_time_mean': terminal_velocity['terminal_tx_time_mean'].mean(),
            'terminal_tx_time_std': terminal_velocity['terminal_tx_time_std'].mean()
        })
        
        df = df.merge(terminal_velocity, on='TERMINAL_ID', how='left')
        
        # Fraud rate at terminal with smoothing
        terminal_fraud = df.groupby('TERMINAL_ID').agg({
            'TX_FRAUD': [
                ('terminal_fraud_rate', 'mean'),
                ('terminal_fraud_count', 'sum')
            ]
        }).reset_index()
        
        # Flatten column names
        terminal_fraud.columns = ['TERMINAL_ID', 'terminal_fraud_rate', 'terminal_fraud_count']
        
        # Apply Laplace smoothing to fraud rate
        alpha = 0.01  # smoothing parameter
        total_transactions = df.groupby('TERMINAL_ID').size().reset_index(name='total_tx')
        terminal_fraud = terminal_fraud.merge(total_transactions, on='TERMINAL_ID', how='left')
        terminal_fraud['terminal_fraud_rate_smoothed'] = (
            (terminal_fraud['terminal_fraud_count'] + alpha) / 
            (terminal_fraud['total_tx'] + 2 * alpha)
        )
        
        df = df.merge(terminal_fraud[['TERMINAL_ID', 'terminal_fraud_rate_smoothed']], 
                     on='TERMINAL_ID', how='left')
        df['terminal_fraud_rate_smoothed'] = df['terminal_fraud_rate_smoothed'].fillna(0)
        
        return df
    
    def _create_sequence_features(self, df):
        """Create sequence-based features with null handling"""
        # Sort transactions by customer and time
        df = df.sort_values(['CUSTOMER_ID', 'TX_DATETIME']).copy()
        
        # Time since last transaction
        df['time_since_last'] = df.groupby('CUSTOMER_ID')['TX_DATETIME'].diff().dt.total_seconds()
        df['time_since_last'] = df['time_since_last'].fillna(0)
        
        # Time until next transaction
        df['time_until_next'] = df.groupby('CUSTOMER_ID')['TX_DATETIME'].diff(-1).dt.total_seconds()
        df['time_until_next'] = df['time_until_next'].fillna(0)
        
        # Amount sequence features
        df['amount_diff_last'] = df.groupby('CUSTOMER_ID')['TX_AMOUNT'].diff()
        df['amount_diff_next'] = df.groupby('CUSTOMER_ID')['TX_AMOUNT'].diff(-1)
        df['amount_diff_last'] = df['amount_diff_last'].fillna(0)
        df['amount_diff_next'] = df['amount_diff_next'].fillna(0)
        
        # Terminal sequence features
        df['terminal_changed'] = (
            df.groupby('CUSTOMER_ID')['TERMINAL_ID'].shift() != df['TERMINAL_ID']
        ).astype(int)
        df['terminal_changed'] = df['terminal_changed'].fillna(1)  # First transaction
        
        # Transaction velocity features using time windows
        def count_transactions(group):
            windows_1h, windows_24h = pd.Timedelta('1h'), pd.Timedelta('24h')
            tx_counts_1h, tx_counts_24h, amount_sums_1h, amount_sums_24h, unique_terminals_24h = [], [], [], [], []

            for idx, row in group.iterrows():
                current_time = row['TX_DATETIME']
                mask_1h = (group['TX_DATETIME'] <= current_time) & (group['TX_DATETIME'] > current_time - windows_1h)
                mask_24h = (group['TX_DATETIME'] <= current_time) & (group['TX_DATETIME'] > current_time - windows_24h)

                tx_counts_1h.append(mask_1h.sum())
                amount_sums_1h.append(group.loc[mask_1h, 'TX_AMOUNT'].sum())
                tx_counts_24h.append(mask_24h.sum())
                amount_sums_24h.append(group.loc[mask_24h, 'TX_AMOUNT'].sum())
                unique_terminals_24h.append(group.loc[mask_24h, 'TERMINAL_ID'].nunique())

            return pd.DataFrame({
                'tx_velocity_1h': tx_counts_1h,
                'tx_velocity_24h': tx_counts_24h,
                'amount_velocity_1h': amount_sums_1h,
                'amount_velocity_24h': amount_sums_24h,
                'unique_terminals_24h': unique_terminals_24h
            }, index=group.index)

        # Apply the velocity calculations and reset index for merging
        velocity_features = df.groupby('CUSTOMER_ID', group_keys=False).apply(count_transactions)

        # Merge velocity features with the original dataframe
        df = df.merge(velocity_features, left_index=True, right_index=True)

        # Sequential pattern features
        df['repeated_terminal'] = df.groupby(['CUSTOMER_ID', 'TERMINAL_ID']).cumcount()

        # Fill any remaining NaN values in sequence columns
        sequence_columns = [
            'tx_velocity_1h', 'tx_velocity_24h',
            'amount_velocity_1h', 'amount_velocity_24h',
            'unique_terminals_24h', 'repeated_terminal'
        ]
        df[sequence_columns] = df[sequence_columns].fillna(0)

        return df
    
    def transform(self, df, training=True):
        """Main transformation pipeline with missing value handling"""
        # Clean data
        df = self._clean_data(df)
        
        # Extract datetime features
        df = self._extract_datetime_features(df)
        
        # Create feature groups
        df = self._create_amount_features(df)
        df = self._create_customer_features(df)
        df = self._create_terminal_features(df)
        df = self._create_sequence_features(df)
        
        # Define feature groups
        feature_groups = self.feature_groups

        # Impute missing values
        imputed_groups = self._impute_missing_values(df, feature_groups)
        
        # Scale features
        scaled_features = {}
        for group, data in imputed_groups.items():
            if group in ['amount', 'sequence']:
                scaled_features[group] = self.robust_scaler.fit_transform(data)
            else:
                scaled_features[group] = self.standard_scaler.fit_transform(data)
        
        if training:
            # Apply SMOTE only during training
            smote = SMOTE(random_state=42)
            
            # Combine all scaled features
            X = np.concatenate([scaled_features[group] for group in feature_groups.keys()], axis=1)
            y = df['TX_FRAUD'].values
            
            # Verify no NaN values before SMOTE
            if np.isnan(X).any():
                print("Warning: NaN values found after preprocessing. Filling with 0...")
                X = np.nan_to_num(X, 0)
            
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