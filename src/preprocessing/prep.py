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
import pickle as pkl

class TransactionPreprocessor:
    def _extract_datetime_features(self, df):
        """Extract temporal features from TX_DATETIME with enhanced patterns"""
        df = df.copy()
        try:
            df['TX_DATETIME'] = pd.to_datetime(df['TX_DATETIME'])
        except (ValueError, TypeError):
            df['TX_DATETIME'] = pd.to_datetime(df['TX_DATETIME'], errors='coerce')
        
        # Basic time-based features
        df['hour'] = df['TX_DATETIME'].dt.hour
        df['day_of_week'] = df['TX_DATETIME'].dt.dayofweek
        df['month'] = df['TX_DATETIME'].dt.month
        
        # Enhanced temporal patterns
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_night'] = (
            (df['hour'] >= 23) | 
            (df['hour'] <= 4)
        ).astype(int)
        
        # Refined rush hour definition
        morning_rush = (df['hour'] >= 7) & (df['hour'] <= 9)
        evening_rush = (df['hour'] >= 16) & (df['hour'] <= 18)
        lunch_rush = (df['hour'] >= 12) & (df['hour'] <= 13)
        df['is_rush_hour'] = (
            morning_rush | 
            evening_rush | 
            lunch_rush
        ).astype(int)
        
        # Holiday features
        df['is_holiday'] = df['TX_DATETIME'].apply(
            lambda x: x in self.us_holidays if pd.notnull(x) else False
        ).astype(int)
        
        # Additional temporal risk factors
        df['is_month_start'] = (df['TX_DATETIME'].dt.day <= 3).astype(int)
        df['is_month_end'] = (df['TX_DATETIME'].dt.day >= 28).astype(int)
        df['is_payday'] = ((df['TX_DATETIME'].dt.day == 15) | 
                          (df['TX_DATETIME'].dt.day == 30) |
                          (df['TX_DATETIME'].dt.day == 31)).astype(int)
        
        return df
    
    def _create_amount_features(self, df):
        """Create amount-based features with enhanced risk patterns"""
        df = df.copy()
        
        # Basic amount features
        df['TX_AMOUNT'] = df['TX_AMOUNT'].clip(lower=0)
        df['amount_log'] = np.log1p(df['TX_AMOUNT'])
        
        # Customer amount statistics with rolling windows
        amount_stats = df.groupby('CUSTOMER_ID').agg({
            'TX_AMOUNT': ['mean', 'std', 'max', 'min']
        }).fillna(0)
        amount_stats.columns = ['mean', 'std', 'max', 'min']
        df = df.merge(amount_stats, on='CUSTOMER_ID', how='left')
        
        # Handle zero standard deviation
        df['std'] = df['std'].replace(0, df['std'].mean())
        
        # Enhanced amount patterns
        df['amount_deviation'] = abs(df['TX_AMOUNT'] - df['mean']) / df['std']
        df['amount_to_mean_ratio'] = df['TX_AMOUNT'] / (df['mean'] + 1)
        
        # Rolling amount statistics
        for window in ['1D', '7D', '30D']:
            df[f'amount_mean_{window}'] = df.groupby('CUSTOMER_ID')['TX_AMOUNT'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            df[f'amount_std_{window}'] = df.groupby('CUSTOMER_ID')['TX_AMOUNT'].transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )
        
        # Amount percentile features
        df['amount_percentile'] = df.groupby('CUSTOMER_ID')['TX_AMOUNT'].transform(
            lambda x: pd.qcut(x, q=10, labels=False, duplicates='drop')
        )
        
        return df
    
    def _create_customer_features(self, df):
        """Create customer-related features with behavioral patterns"""
        df = df.copy()
        
        # Basic customer statistics
        customer_stats = df.groupby('CUSTOMER_ID').agg({
            'TX_DATETIME': 'count',
            'TERMINAL_ID': 'nunique',
            'hour': ['mean', 'std']
        }).fillna(0)
        
        customer_stats.columns = [
            'customer_tx_count', 
            'customer_terminal_count',
            'customer_hour_mean', 
            'customer_hour_std'
        ]
        
        # Enhanced customer behavior patterns
        customer_enhanced = df.groupby('CUSTOMER_ID').agg({
            'TX_AMOUNT': lambda x: np.percentile(x, 75),  # High spend threshold
            'is_weekend': 'mean',  # Weekend transaction ratio
            'is_night': 'mean',    # Night transaction ratio
            'terminal_changed': 'mean'  # Terminal switching ratio
        }).fillna(0)
        
        customer_enhanced.columns = [
            'customer_high_spend_threshold',
            'customer_weekend_ratio',
            'customer_night_ratio',
            'customer_terminal_switching_ratio'
        ]
        
        # Merge all customer features
        df = df.merge(customer_stats, on='CUSTOMER_ID', how='left')
        df = df.merge(customer_enhanced, on='CUSTOMER_ID', how='left')
        
        # Add customer risk scoring
        risk_patterns = df.groupby('CUSTOMER_ID').agg({
            'amount_deviation': 'mean',
            'terminal_recent_fraud_rate': 'mean',
            'amount_velocity_ratio': 'mean'
        }).fillna(0)
        
        risk_patterns.columns = [
            'customer_amount_risk',
            'customer_terminal_risk',
            'customer_velocity_risk'
        ]
        
        df = df.merge(risk_patterns, on='CUSTOMER_ID', how='left')
        
        return df
    
    def _create_terminal_features(self, df):
        """Create terminal-related features with risk patterns"""
        df = df.copy()
        
        # Basic terminal statistics
        terminal_stats = df.groupby('TERMINAL_ID').agg({
            'TX_AMOUNT': ['count', 'mean', 'std', 'median'],
            'TX_DATETIME': lambda x: x.diff().mean().total_seconds() if len(x) > 1 else 0
        }).fillna(0)
        
        terminal_stats.columns = [
            'terminal_tx_count', 
            'terminal_amount_mean',
            'terminal_amount_std', 
            'terminal_amount_median',
            'terminal_tx_time_mean'
        ]
        
        # Enhanced terminal patterns
        terminal_enhanced = df.groupby('TERMINAL_ID').agg({
            'TX_AMOUNT': lambda x: (x > x.mean() + 2*x.std()).sum(),  # Suspicious transactions
            'customer_tx_count': 'nunique',  # Unique customers
            'is_night': 'mean',  # Night transaction ratio
            'is_weekend': 'mean'  # Weekend transaction ratio
        }).fillna(0)
        
        terminal_enhanced.columns = [
            'terminal_suspicious_tx_count',
            'terminal_unique_customers',
            'terminal_night_ratio',
            'terminal_weekend_ratio'
        ]
        
        # Terminal fraud patterns
        recent_window = self.feature_params['recent_fraud_window']
        terminal_risk = df.set_index('TX_DATETIME').groupby('TERMINAL_ID')['TX_FRAUD'].rolling(
            recent_window
        ).agg(['mean', 'sum']).reset_index()
        
        terminal_risk.columns = [
            'TX_DATETIME', 
            'TERMINAL_ID',
            'terminal_recent_fraud_rate',
            'terminal_recent_fraud_count'
        ]
        
        # Terminal volatility patterns
        volatility_span = self.feature_params['volatility_span']
        df['terminal_amount_volatility'] = df.groupby('TERMINAL_ID')['TX_AMOUNT'].transform(
            lambda x: x.ewm(span=volatility_span).std()
        )
        
        # Merge all terminal features
        df = df.merge(terminal_stats, on='TERMINAL_ID', how='left')
        df = df.merge(terminal_enhanced, on='TERMINAL_ID', how='left')
        df = df.merge(terminal_risk[['TERMINAL_ID', 'terminal_recent_fraud_rate']], 
                     on='TERMINAL_ID', 
                     how='left')
        
        return df
    
    def _create_sequence_features(self, df):
        """Create sequence-related features with behavioral patterns"""
        df = df.copy()
        df = df.sort_values(['CUSTOMER_ID', 'TX_DATETIME'])
        
        # Time-based sequence features
        df['time_since_last'] = df.groupby('CUSTOMER_ID')['TX_DATETIME'].diff().dt.total_seconds().fillna(0)
        df['time_until_next'] = df.groupby('CUSTOMER_ID')['TX_DATETIME'].diff(-1).dt.total_seconds().fillna(0)
        
        # Amount sequence features
        df['amount_diff_last'] = df.groupby('CUSTOMER_ID')['TX_AMOUNT'].diff().fillna(0)
        df['amount_diff_next'] = df.groupby('CUSTOMER_ID')['TX_AMOUNT'].diff(-1).fillna(0)
        
        # Terminal sequence features
        df['terminal_changed'] = (
            df.groupby('CUSTOMER_ID')['TERMINAL_ID'].shift() != df['TERMINAL_ID']
        ).astype(int).fillna(1)
        df['repeated_terminal'] = df.groupby(['CUSTOMER_ID', 'TERMINAL_ID']).cumcount()
        
        # Enhanced velocity features
        short_window = self.feature_params['velocity_windows']['short']
        long_window = self.feature_params['velocity_windows']['long']
        
        for window in [short_window, long_window]:
            # Transaction velocity
            df[f'tx_velocity_{window}'] = df.groupby('CUSTOMER_ID').rolling(
                window=window,
                on='TX_DATETIME',
                min_periods=1
            ).size().reset_index(0, drop=True)
            
            # Amount velocity
            df[f'amount_velocity_{window}'] = df.groupby('CUSTOMER_ID')['TX_AMOUNT'].rolling(
                window=window,
                min_periods=1
            ).sum().reset_index(0, drop=True)
            
            # Terminal velocity
            df[f'terminal_velocity_{window}'] = df.groupby('CUSTOMER_ID')['TERMINAL_ID'].rolling(
                window=window,
                min_periods=1
            ).nunique().reset_index(0, drop=True)
        
        # Velocity ratios
        df['velocity_ratio_1h_24h'] = df['tx_velocity_1h'] / (df['tx_velocity_24h'] + 1)
        df['amount_velocity_ratio'] = df['amount_velocity_1h'] / (df['amount_velocity_24h'] + 1)
        df['terminal_velocity_ratio'] = df['terminal_velocity_1h'] / (df['terminal_velocity_24h'] + 1)
        
        # Terminal switching patterns
        df['terminal_switches_1h'] = df.groupby('CUSTOMER_ID').apply(
            lambda x: x['terminal_changed'].rolling(short_window, on='TX_DATETIME').sum()
        ).reset_index(level=0, drop=True)
        
        return df

    def transform(self, df, training=True):
        """Transform data with consistent feature handling and debugging"""
        print("Starting feature transformation...")
        df = df.copy()
        
        # Create all features with progress tracking
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
        
        # Process each feature group
        transformed_groups = {}
        for group, features in self.feature_groups.items():
            print(f"Processing {group} features...")
            
            # Extract and order features
            group_data = df[features].copy()
            
            # Debug information
            missing_cols = set(features) - set(df.columns)
            if missing_cols:
                print(f"Warning: Missing columns in {group}: {missing_cols}")
            
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
            
            # Quality checks
            if np.isnan(transformed_groups[group]).any():
                print(f"Warning: NaN values detected in {group} after transformation")
            
        if training:
            # Combine features for SMOTE
            X = np.concatenate([transformed_groups[group] for group in self.feature_groups.keys()], axis=1)
            y = df['TX_FRAUD'].values
            
            # Print class distribution
            print("\nClass distribution before SMOTE:")
            print(f"Non-fraud: {(y == 0).sum()}, Fraud: {(y == 1).sum()}")
            
            # Apply SMOTE with careful sampling
            n_minority = (y == 1).sum()
            n_majority = (y == 0).sum()
            sampling_ratio = min(0.15, n_majority / (n_minority * 3))  # Conservative ratio
            
            smote = SMOTE(
                sampling_strategy=sampling_ratio,
                random_state=42,
                k_neighbors=min(5, n_minority - 1),
                n_jobs=1
            )
            
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            print("\nClass distribution after SMOTE:")
            print(f"Non-fraud: {(y_resampled == 0).sum()}, Fraud: {(y_resampled == 1).sum()}")
            
            # Split back into feature groups
            resampled_groups = {}
            start_idx = 0
            for group in self.feature_groups.keys():
                end_idx = start_idx + len(self.feature_groups[group])
                resampled_groups[group] = X_resampled[:, start_idx:end_idx]
                start_idx = end_idx
            
            # Verify dimensions
            total_features = sum(len(self.feature_groups[g]) for g in self.feature_groups)
            if start_idx != total_features:
                print(f"Warning: Feature dimension mismatch. Expected {total_features}, got {start_idx}")
            
            return resampled_groups, y_resampled
        
        # Verify and clean transformed groups
        for group in transformed_groups:
            # Replace any remaining infinities with large values
            transformed_groups[group] = np.nan_to_num(
                transformed_groups[group], 
                posinf=1e10, 
                neginf=-1e10
            )
        
        return transformed_groups

    def get_feature_importance_groups(self):
        """Returns a dictionary mapping features to their groups for interpretation"""
        feature_groups_map = {}
        for group, features in self.feature_groups.items():
            for feature in features:
                feature_groups_map[feature] = group
        return feature_groups_map

    def generate_feature_report(self, df):
        """Generate a comprehensive report on feature statistics and quality"""
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
                    'min': group_data[feature].min(),
                    'max': group_data[feature].max(),
                    'mean': group_data[feature].mean(),
                    'std': group_data[feature].std()
                } for feature in features
            }
            
            # Correlation analysis within group
            if len(features) > 1:
                corr_matrix = group_data.corr()
                high_corr = np.where(np.abs(corr_matrix) > 0.8)
                high_corr_pairs = set()
                for i, j in zip(*high_corr):
                    if i != j and corr_matrix.index[i] < corr_matrix.columns[j]:
                        high_corr_pairs.add(
                            (corr_matrix.index[i], 
                             corr_matrix.columns[j], 
                             corr_matrix.iloc[i, j])
                        )
                report['correlations'][group] = list(high_corr_pairs)
        
        return report

    def save_preprocessor(self, filepath):
        """Save the preprocessor state to a file"""
        save_dict = {
            'scalers': self.scalers,
            'imputers': self.imputers,
            'feature_groups': self.feature_groups,
            'feature_params': self.feature_params
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
        preprocessor.feature_params = save_dict['feature_params']
        
        return preprocessor

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
                issues.append("TX_DATETIME is not in datetime format")
        
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
        
        return issues

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
