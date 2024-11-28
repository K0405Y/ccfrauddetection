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
import warnings
warnings.filterwarnings('ignore')

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

    def _safe_datetime_conversion(self, series):
        """Safely convert datetime with multiple format attempts"""
        common_formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y/%m/%d %H:%M:%S',
            '%d-%m-%Y %H:%M:%S',
            '%m/%d/%Y %H:%M:%S'
        ]
        
        for fmt in common_formats:
            try:
                return pd.to_datetime(series, format=fmt)
            except ValueError:
                continue
        
        # If no format works, try coercing
        result = pd.to_datetime(series, errors='coerce')
        null_count = result.isnull().sum()
        if null_count > 0:
            warnings.warn(f"Could not parse {null_count} datetime values")
        return result

    def _extract_datetime_features(self, df):
        """Extract temporal features from TX_DATETIME with enhanced error handling"""
        df = df.copy()
        try:
            df['TX_DATETIME'] = self._safe_datetime_conversion(df['TX_DATETIME'])
        except Exception as e:
            print(f"Error in datetime conversion: {str(e)}")
            raise
            
        # Time-based features with null handling
        df['hour'] = df['TX_DATETIME'].dt.hour
        df['day_of_week'] = df['TX_DATETIME'].dt.dayofweek
        df['month'] = df['TX_DATETIME'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_night'] = ((df['hour'] >= 23) | (df['hour'] <= 4)).astype(int)
        df['is_rush_hour'] = (((df['hour'] >= 8) & (df['hour'] <= 10)) |
                          ((df['hour'] >= 16) & (df['hour'] <= 18))).astype(int)
        df['is_holiday'] = df['TX_DATETIME'].apply(
            lambda x: x in self.us_holidays if pd.notnull(x) else False).astype(int)
        
        # Fill any remaining nulls with median
        temporal_features = self.feature_groups['temporal']
        for feature in temporal_features:
            if df[feature].isnull().any():
                df[feature] = df[feature].fillna(df[feature].median())
            
        return df

    def _create_amount_features(self, df):
        """Create amount-based features with improved handling of edge cases"""
        df = df.copy()
        
        # Ensure positive amounts
        df['TX_AMOUNT'] = df['TX_AMOUNT'].clip(lower=0)
        df['amount_log'] = np.log1p(df['TX_AMOUNT'])
        
        # Customer amount statistics with careful null handling
        amount_stats = df.groupby('CUSTOMER_ID')['TX_AMOUNT'].agg(
            ['mean', 'std', 'max', 'min']
        ).apply(lambda x: x.fillna(x.median()))
        
        # Handle zero std cases
        min_std = amount_stats['std'].mean() * 0.01  # Use 1% of mean std as minimum
        amount_stats['std'] = amount_stats['std'].replace(0, min_std)
        
        df = df.merge(amount_stats, on='CUSTOMER_ID', how='left')
        
        # Calculate amount deviation with safe division
        df['amount_deviation'] = abs(df['TX_AMOUNT'] - df['mean']) / df['std'].clip(lower=min_std)
        
        # Fill any remaining nulls with median
        amount_features = self.feature_groups['amount']
        for feature in amount_features:
            if df[feature].isnull().any():
                df[feature] = df[feature].fillna(df[feature].median())
        
        return df

    def _create_customer_features(self, df):
        """Create customer-related features with null handling"""
        df = df.copy()
        
        # Transaction counts with null handling
        customer_stats = df.groupby('CUSTOMER_ID').agg({
            'TX_DATETIME': 'count',
            'TERMINAL_ID': 'nunique',
            'hour': ['mean', 'std']
        }).apply(lambda x: x.fillna(x.median()))
        
        customer_stats.columns = [
            'customer_tx_count', 'customer_terminal_count',
            'customer_hour_mean', 'customer_hour_std'
        ]
        
        df = df.merge(customer_stats, on='CUSTOMER_ID', how='left')
        
        # Fill any remaining nulls with median
        customer_features = self.feature_groups['customer']
        for feature in customer_features:
            if df[feature].isnull().any():
                df[feature] = df[feature].fillna(df[feature].median())
        
        return df

    def _create_terminal_features(self, df):
        """Create terminal-related features with improved handling"""
        df = df.copy()
        
        # Terminal transaction statistics with careful null handling
        terminal_stats = df.groupby('TERMINAL_ID').agg({
            'TX_AMOUNT': ['count', 'mean', 'std', 'median'],
            'TX_DATETIME': lambda x: x.diff().mean().total_seconds() if len(x) > 1 else 0
        }).apply(lambda x: x.fillna(x.median()))
        
        terminal_stats.columns = [
            'terminal_tx_count', 'terminal_amount_mean',
            'terminal_amount_std', 'terminal_amount_median',
            'terminal_tx_time_mean'
        ]
        
        # Additional terminal features
        terminal_stats['terminal_tx_count_large'] = df.groupby('TERMINAL_ID')['TX_AMOUNT'].apply(
            lambda x: (x > x.mean()).sum()
        ).fillna(0)
        
        terminal_stats['terminal_tx_time_std'] = df.groupby('TERMINAL_ID')['TX_DATETIME'].apply(
            lambda x: x.diff().std().total_seconds() if len(x) > 1 else 0
        ).fillna(0)
        
        # Terminal fraud rate with Laplace smoothing
        if 'TX_FRAUD' in df.columns:
            terminal_fraud = df.groupby('TERMINAL_ID')['TX_FRAUD'].agg(['mean', 'sum']).fillna(0)
            terminal_stats['terminal_fraud_rate_smoothed'] = (
                terminal_fraud['sum'] + 0.01
            ) / (terminal_stats['terminal_tx_count'] + 0.02)
        else:
            terminal_stats['terminal_fraud_rate_smoothed'] = 0
        
        df = df.merge(terminal_stats, on='TERMINAL_ID', how='left')
        
        # Fill any remaining nulls with median
        terminal_features = self.feature_groups['terminals']
        for feature in terminal_features:
            if df[feature].isnull().any():
                df[feature] = df[feature].fillna(df[feature].median())
        
        return df

    def _safe_sequence_features(self, df):
        """Memory-efficient sequence feature creation with improved handling"""
        df = df.copy()
        df = df.sort_values(['CUSTOMER_ID', 'TX_DATETIME'])
        
        # Pre-allocate arrays for results
        sequence_features = {
            'time_since_last': np.nan,
            'time_until_next': np.nan,
            'amount_diff_last': np.nan,
            'amount_diff_next': np.nan,
            'terminal_changed': 1,  # Default to 1 for first transaction
            'repeated_terminal': 0,
            'tx_velocity_1h': 0,
            'tx_velocity_24h': 0,
            'amount_velocity_1h': 0,
            'amount_velocity_24h': 0,
            'unique_terminals_24h': 0
        }
        
        for feature, default in sequence_features.items():
            df[feature] = default
        
        # Process in chunks for memory efficiency
        chunk_size = 10000
        for start_idx in range(0, len(df), chunk_size):
            end_idx = min(start_idx + chunk_size, len(df))
            chunk = df.iloc[start_idx:end_idx]
            
            # Calculate basic sequence features
            group = chunk.groupby('CUSTOMER_ID')
            df.loc[chunk.index, 'time_since_last'] = group['TX_DATETIME'].diff().dt.total_seconds()
            df.loc[chunk.index, 'time_until_next'] = group['TX_DATETIME'].diff(-1).dt.total_seconds()
            df.loc[chunk.index, 'amount_diff_last'] = group['TX_AMOUNT'].diff()
            df.loc[chunk.index, 'amount_diff_next'] = group['TX_AMOUNT'].diff(-1)
            
            # Terminal changes
            df.loc[chunk.index, 'terminal_changed'] = (
                group['TERMINAL_ID'].shift() != chunk['TERMINAL_ID']
            ).astype(int)
            
            # Calculate velocity features
            for window in ['1h', '24h']:
                td = pd.Timedelta(window)
                for idx in chunk.index:
                    current_time = df.loc[idx, 'TX_DATETIME']
                    customer_mask = (
                        (df['CUSTOMER_ID'] == df.loc[idx, 'CUSTOMER_ID']) &
                        (df['TX_DATETIME'] <= current_time) &
                        (df['TX_DATETIME'] > current_time - td)
                    )
                    
                    df.loc[idx, f'tx_velocity_{window}'] = customer_mask.sum()
                    df.loc[idx, f'amount_velocity_{window}'] = df.loc[customer_mask, 'TX_AMOUNT'].sum()
                    
                    if window == '24h':
                        df.loc[idx, 'unique_terminals_24h'] = df.loc[customer_mask, 'TERMINAL_ID'].nunique()
        
        # Fill nulls with median for all sequence features
        sequence_features = self.feature_groups['sequence']
        for feature in sequence_features:
            if df[feature].isnull().any():
                df[feature] = df[feature].fillna(df[feature].median())
        
        return df

    def _create_features(self, df):
        """Create all features with improved error handling and progress tracking"""
        df = df.copy()
        
        try:
            print("Extracting datetime features...")
            df = self._extract_datetime_features(df)
            
            print("Creating amount features...")
            df = self._create_amount_features(df)
            
            print("Creating customer features...")
            df = self._create_customer_features(df)
            
            print("Creating terminal features...")
            df = self._create_terminal_features(df)
            
            print("Creating sequence features...")
            df = self._safe_sequence_features(df)
            
            # Ensure all features exist and handle any remaining nulls
            print("Validating features...")
            all_features = self.get_feature_names()
            missing = set(all_features) - set(df.columns)
            if missing:
                raise ValueError(f"Failed to create features: {missing}")
            
            # Final null check across all features
            for feature in all_features:
                if df[feature].isnull().any():
                    print(f"Warning: Filling remaining nulls in {feature}")
                    df[feature] = df[feature].fillna(df[feature].median())
            
        except Exception as e:
            print(f"Error in feature creation: {str(e)}")
            raise
        
        return df

    def transform(self, df, training=True):
        """Transform data with consistent feature handling and improved error checking"""
        print("Starting feature transformation...")
        df = df.copy()
        
        try:
            # Create features
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
                
                print(f"\nTraining Data Class distribution before SMOTE:")
                print(f"Non-fraud: {(y == 0).sum()}, Fraud: {(y == 1).sum()}")
                
                # Apply SMOTE with careful sampling
                n_minority = (y == 1).sum()
                n_majority = (y == 0).sum()
                target_minority = int(n_majority * 0.20)  # 20% fraud ratio
                
                smote = SMOTE(
                    sampling_strategy={1: min(target_minority, n_majority)},
                    random_state=0,
                    k_neighbors=min(5, n_minority - 1),
                    n_jobs=1
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
            
        except Exception as e:
            print(f"Error in transform: {str(e)}")
            raise

    def get_feature_names(self):
        """Return a list of all feature names across all feature groups"""
        all_features = []
        for features in self.feature_groups.values():
            all_features.extend(features)
        return all_features

    def get_feature_metadata(self):
        """Return metadata about features for documentation"""
        metadata = {}
        for group, features in self.feature_groups.items():
            metadata[group] = {
                'feature_count': len(features),
                'features': features,
                'scaler_type': type(self.scalers[group]).__name__,
                'imputer_strategy': self.imputers[group].strategy
            }
        return metadata

    def validate_features(self, df):
        """Validate features and return any issues found"""
        issues = []
        
        # Check for required columns
        required_columns = ['TX_DATETIME', 'CUSTOMER_ID', 'TERMINAL_ID', 'TX_AMOUNT']
        missing_required = set(required_columns) - set(df.columns)
        if missing_required:
            issues.append(f"Missing required columns: {missing_required}")
        
        # Check data types
        if 'TX_DATETIME' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['TX_DATETIME']):
                try:
                    self._safe_datetime_conversion(df['TX_DATETIME'])
                except:
                    issues.append("TX_DATETIME cannot be converted to datetime format")
        
        # Check for negative amounts
        if 'TX_AMOUNT' in df.columns:
            if (df['TX_AMOUNT'] < 0).any():
                issues.append("Negative values found in TX_AMOUNT")
        
        # Check for duplicate transactions
        duplicates = df.duplicated(subset=['TX_DATETIME', 'CUSTOMER_ID', 'TERMINAL_ID', 'TX_AMOUNT'])
        if duplicates.any():
            issues.append(f"Found {duplicates.sum()} duplicate transactions")
        
        # Verify ID formats
        if 'CUSTOMER_ID' in df.columns:
            if df['CUSTOMER_ID'].isnull().any():
                issues.append("Missing values in CUSTOMER_ID")
        
        if 'TERMINAL_ID' in df.columns:
            if df['TERMINAL_ID'].isnull().any():
                issues.append("Missing values in TERMINAL_ID")
        
        # Check if fraud label exists for training
        if 'TX_FRAUD' not in df.columns:
            issues.append("TX_FRAUD column missing - required for training")
        elif not df['TX_FRAUD'].isin([0, 1]).all():
            issues.append("TX_FRAUD contains values other than 0 and 1")
        
        return issues