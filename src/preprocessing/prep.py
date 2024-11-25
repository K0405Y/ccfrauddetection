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
import pickle as pkl
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

    def _extract_datetime_features(self, df):
        """Extract temporal features from TX_DATETIME"""
        df = df.copy()
        try:
            df['TX_DATETIME'] = pd.to_datetime(df['TX_DATETIME'])
        except (ValueError, TypeError):
            print("Warning: Error converting TX_DATETIME. Attempting to coerce...")
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
        amount_stats = df.groupby('CUSTOMER_ID')['TX_AMOUNT'].agg(
            ['mean', 'std', 'max', 'min']
        ).fillna(0)
        
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
        }).apply(lambda x: x.fillna(x.median()))

        
        customer_stats.columns = [
            'customer_tx_count', 'customer_terminal_count',
            'customer_hour_mean', 'customer_hour_std'
        ]
        
        df = df.merge(customer_stats, on='CUSTOMER_ID', how='left')
        return df

    def _create_terminal_features(self, df):
        """Create terminal-related features"""
        df = df.copy()
        
        # Terminal transaction statistics
        terminal_stats = df.groupby('TERMINAL_ID').agg({
            'TX_AMOUNT': ['count', 'mean', 'std', 'median'],
            'TX_DATETIME': lambda x: x.diff().mean().total_seconds() if len(x) > 1 else 0
        })
        
        # Fill missing values with the median of each column
        terminal_stats = terminal_stats.apply(lambda x: x.fillna(x.median()))
        
        terminal_stats.columns = [
            'terminal_tx_count', 'terminal_amount_mean',
            'terminal_amount_std', 'terminal_amount_median',
            'terminal_tx_time_mean'
        ]
        
        # Additional terminal features
        terminal_stats['terminal_tx_count_large'] = df.groupby('TERMINAL_ID')['TX_AMOUNT'].apply(
            lambda x: (x > x.mean()).sum()
        ).apply(lambda x: x.fillna(x.median()))
        
        terminal_stats['terminal_tx_time_std'] = df.groupby('TERMINAL_ID')['TX_DATETIME'].apply(
            lambda x: x.diff().std().total_seconds() if len(x) > 1 else 0
        ).apply(lambda x: x.fillna(x.median()))
        
        # Terminal fraud rate with Laplace smoothing
        if 'TX_FRAUD' in df.columns:
            terminal_fraud = df.groupby('TERMINAL_ID')['TX_FRAUD'].agg(['mean', 'sum']).apply(lambda x: x.fillna(x.median()))
            terminal_stats['terminal_fraud_rate_smoothed'] = (
                terminal_fraud['sum'] + 0.01
            ) / (terminal_stats['terminal_tx_count'] + 0.02)
        else:
            terminal_stats['terminal_fraud_rate_smoothed'] = 0
        
        df = df.merge(terminal_stats, on='TERMINAL_ID', how='left')
        return df
        
    def _create_sequence_features(self, df):
        """Create sequence-related features"""
        df = df.copy()
        df = df.sort_values(['CUSTOMER_ID', 'TX_DATETIME'])
        
        # Time-based sequence features
        df['time_since_last'] = df.groupby('CUSTOMER_ID')['TX_DATETIME'].diff().dt.total_seconds().apply(lambda x: x.fillna(x.median()))

        df['time_until_next'] = df.groupby('CUSTOMER_ID')['TX_DATETIME'].diff(-1).dt.total_seconds().apply(lambda x: x.fillna(x.median()))
        
        # Amount sequence features
        df['amount_diff_last'] = df.groupby('CUSTOMER_ID')['TX_AMOUNT'].diff().apply(lambda x: x.fillna(x.median()))
        df['amount_diff_next'] = df.groupby('CUSTOMER_ID')['TX_AMOUNT'].diff(-1).apply(lambda x: x.fillna(x.median()))
        
        # Terminal sequence features
        df['terminal_changed'] = (
            df.groupby('CUSTOMER_ID')['TERMINAL_ID'].shift() != df['TERMINAL_ID']
        ).astype(int).fillna(1)
        df['repeated_terminal'] = df.groupby(['CUSTOMER_ID', 'TERMINAL_ID']).cumcount()
        
        # Velocity features
        for window in ['1h', '24h']:
            td = pd.Timedelta(window)
            
            # Initialize velocity columns
            df[f'tx_velocity_{window}'] = 0
            df[f'amount_velocity_{window}'] = 0
            if window == '24h':
                df['unique_terminals_24h'] = 0
            
            for customer_id in df['CUSTOMER_ID'].unique():
                mask = df['CUSTOMER_ID'] == customer_id
                customer_data = df[mask].copy()
                
                # Calculate rolling counts and amounts
                for idx in customer_data.index:
                    time = customer_data.loc[idx, 'TX_DATETIME']
                    window_mask = (
                        (customer_data['TX_DATETIME'] <= time) & 
                        (customer_data['TX_DATETIME'] > time - td)
                    )
                    
                    df.loc[idx, f'tx_velocity_{window}'] = window_mask.sum()
                    df.loc[idx, f'amount_velocity_{window}'] = customer_data.loc[window_mask, 'TX_AMOUNT'].sum()
                    
                    if window == '24h':
                        df.loc[idx, 'unique_terminals_24h'] = customer_data.loc[window_mask, 'TERMINAL_ID'].nunique()
        
        return df

    def _create_features(self, df):
        """Create all features in consistent order"""
        df = df.copy()
        
        # Create features with progress tracking
        print("Extracting datetime features...")
        df = self._extract_datetime_features(df)
        
        print("Creating amount features...")
        df = self._create_amount_features(df)
        
        print("Creating customer features...")
        df = self._create_customer_features(df)
        
        print("Creating terminal features...")
        df = self._create_terminal_features(df)
        
        print("Creating sequence features...")
        df = self._create_sequence_features(df)
        
        # Ensure all features exist with correct ordering
        print("Validating features...")
        for group, features in self.feature_groups.items():
            missing_features = set(features) - set(df.columns)
            if missing_features:
                print(f"Warning: Creating missing features for group {group}: {missing_features}")
            for feature in missing_features:
                df[feature] = 0
        
        # Verify all required features exist
        all_features = self.get_feature_names()
        missing = set(all_features) - set(df.columns)
        if missing:
            raise ValueError(f"Failed to create features: {missing}")
        
        return df

    def generate_feature_report(self, df):
        """Generate a comprehensive report on feature statistics and quality"""
        # Create features first
        df = self._create_features(df)
        
        report = {
            'feature_counts': {group: len(features) for group, features in self.feature_groups.items()},
            'missing_values': {},
            'unique_counts': {},
            'value_ranges': {},
            'correlations': {}
        }
        
        # Calculate statistics for each feature group
        for group, features in self.feature_groups.items():
            group_data = df[features].copy()
            
            # Missing values analysis
            missing_stats = group_data.isnull().sum()
            report['missing_values'][group] = missing_stats[missing_stats > 0].to_dict()
            
            # Unique values analysis
            report['unique_counts'][group] = group_data.nunique().to_dict()
            
            # Value ranges
            report['value_ranges'][group] = {
                feature: {
                    'min': float(group_data[feature].min()),
                    'max': float(group_data[feature].max()),
                    'mean': float(group_data[feature].mean()),
                    'std': float(group_data[feature].std())
                } for feature in features
            }
            
            # Correlation analysis within group
            if len(features) > 1:
                corr_matrix = group_data.corr()
                high_corr = np.where(np.abs(corr_matrix) > 0.8)
                high_corr_pairs = []
                for i, j in zip(*high_corr):
                    if i != j and corr_matrix.index[i] < corr_matrix.columns[j]:
                        high_corr_pairs.append({
                            'feature1': corr_matrix.index[i],
                            'feature2': corr_matrix.columns[j],
                            'correlation': float(corr_matrix.iloc[i, j])
                        })
                report['correlations'][group] = high_corr_pairs
        
        return report

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
            
            # Apply SMOTE with careful sampling
            n_minority = (y == 1).sum()
            n_majority = (y == 0).sum()
            target_minority = int(n_majority * 0.15)  # 15% fraud ratio
            
            print(f"\nTraining Data Class distribution before SMOTE:")
            print(f"Non-fraud: {n_majority}, Fraud: {n_minority}")
            
            smote = SMOTE(
                sampling_strategy={1: min(target_minority, n_majority)},
                random_state=42,
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
                    pd.to_datetime(df['TX_DATETIME'])
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

    def save_preprocessor(self, filepath):
        """Save the preprocessor state to a file"""
        save_dict = {
            'scalers': self.scalers,
            'imputers': self.imputers,
            'feature_groups': self.feature_groups
        }
        with open(filepath, 'wb') as f:
            pkl.dump(save_dict, f)
    
    @classmethod
    def load_preprocessor(cls, filepath):
        """Load a preprocessor state from a file"""
        with open(filepath, 'rb') as f:
            save_dict = pkl.load(f)
        
        preprocessor = cls()
        preprocessor.scalers = save_dict['scalers']
        preprocessor.imputers = save_dict['imputers']
        preprocessor.feature_groups = save_dict['feature_groups']
        
        return preprocessor