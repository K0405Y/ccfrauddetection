import pandas as pd  
import numpy as np
import datetime  
from scipy.stats import skew, kurtosis
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

def get_train_test_set(transactions_df,
                       start_date_training,
                       delta_train=7, delta_delay=7, delta_test=7,
                       random_state=0):
    """
    Prepare train and test datasets for fraud detection.
    """
    # Convert transaction datetime to pandas datetime
    transactions_df['TX_DATETIME'] = pd.to_datetime(transactions_df['TX_DATETIME'])
    start_date_training = datetime.datetime.strptime(start_date_training, "%Y-%m-%d %H:%M:%S")
    
    # Filter training data based on the provided date range
    train_df = transactions_df[(transactions_df.TX_DATETIME >= start_date_training) &
                                (transactions_df.TX_DATETIME < start_date_training + datetime.timedelta(days=delta_train))]
    
    # Initialize test set as an empty list
    test_df = []
    
    # Identify customers with fraudulent transactions in the training set
    known_defrauded_customers = set(train_df[train_df.TX_FRAUD == 1].CUSTOMER_ID)
    start_tx_time_days_training = train_df.TX_TIME_DAYS.min()  # Training start day
    
    # Loop to construct the test dataset for each day in the test period
    for day in range(delta_test):
        # Get test data for the current day
        test_df_day = transactions_df[transactions_df.TX_TIME_DAYS == start_tx_time_days_training +
                                      delta_train + delta_delay + day]
        
        # Update known defrauded customers based on prior frauds
        test_df_day_delay_period = transactions_df[transactions_df.TX_TIME_DAYS == start_tx_time_days_training +
                                                   delta_train + day - 1]
        new_defrauded_customers = set(test_df_day_delay_period[test_df_day_delay_period.TX_FRAUD == 1].CUSTOMER_ID)
        known_defrauded_customers = known_defrauded_customers.union(new_defrauded_customers)
        
        # Remove transactions involving known defrauded customers from the test set
        test_df_day = test_df_day[~test_df_day.CUSTOMER_ID.isin(known_defrauded_customers)]
        test_df.append(test_df_day)
    
    test_df = pd.concat(test_df)
    
    return train_df, test_df

def is_weekend(tx_datetime):
    """
    Check if a transaction occurred on a weekend.
    """
    return tx_datetime.weekday() >= 5

def is_night(tx_datetime):
    """
    Check if a transaction occurred at night (before 6 AM).
    """
    return int(tx_datetime.hour <= 6)

def get_customer_spending_features(customer_transactions, windows_size_in_days=[1, 7, 30]):
    """
    Calculate customer spending features over defined time windows.
    """
    customer_transactions = customer_transactions.sort_values('TX_DATETIME')  # Sort transactions by date
    customer_transactions.index = customer_transactions.TX_DATETIME  # Set datetime as index
    
    for window_size in windows_size_in_days:
        # Calculate transaction metrics for the window
        SUM_AMOUNT_TX_WINDOW = customer_transactions['TX_AMOUNT'].rolling(window=window_size, min_periods=1).sum()
        NB_TX_WINDOW = customer_transactions['TX_AMOUNT'].rolling(window=window_size, min_periods=1).count()
        AVG_AMOUNT_TX_WINDOW = SUM_AMOUNT_TX_WINDOW / NB_TX_WINDOW
        
        # Add features to the dataset
        customer_transactions[f'CUSTOMER_ID_NB_TX_{window_size}DAY_WINDOW'] = NB_TX_WINDOW
        customer_transactions[f'CUSTOMER_ID_AVG_AMOUNT_{window_size}DAY_WINDOW'] = AVG_AMOUNT_TX_WINDOW
    
    customer_transactions.index = customer_transactions.TRANSACTION_ID  # Reset index to transaction ID
    return customer_transactions

def get_count_risk_rolling_window(terminal_transactions, delay_period=7, windows_size_in_days=[1, 7, 30], feature="TERMINAL_ID"):
    """
    Calculate risk scores for terminals over rolling time windows.
    """
    terminal_transactions = terminal_transactions.sort_values('TX_DATETIME')  # Sort transactions by date
    terminal_transactions.index = terminal_transactions.TX_DATETIME  # Set datetime as index
    
    # Calculate delayed fraud and transaction counts
    NB_FRAUD_DELAY = terminal_transactions['TX_FRAUD'].rolling(f'{delay_period}d').sum()
    NB_TX_DELAY = terminal_transactions['TX_FRAUD'].rolling(f'{delay_period}d').count()
    
    for window_size in windows_size_in_days:
        # Calculate metrics for the rolling window
        NB_FRAUD_DELAY_WINDOW = terminal_transactions['TX_FRAUD'].rolling(f'{delay_period + window_size}d').sum()
        NB_TX_DELAY_WINDOW = terminal_transactions['TX_FRAUD'].rolling(f'{delay_period + window_size}d').count()
        
        NB_FRAUD_WINDOW = NB_FRAUD_DELAY_WINDOW - NB_FRAUD_DELAY
        NB_TX_WINDOW = NB_TX_DELAY_WINDOW - NB_TX_DELAY
        RISK_WINDOW = NB_FRAUD_WINDOW / NB_TX_WINDOW
        
        # Add features to the dataset
        terminal_transactions[f'{feature}_NB_TX_{window_size}DAY_WINDOW'] = list(NB_TX_WINDOW)
        terminal_transactions[f'{feature}_RISK_{window_size}DAY_WINDOW'] = list(RISK_WINDOW)
    
    terminal_transactions.index = terminal_transactions.TRANSACTION_ID  # Reset index to transaction ID

    terminal_transactions.fillna(0, inplace=True)  # Replace NaN values with 0

    return terminal_transactions

def get_network_features(transactions_df, window_size_days=30):
    """
    Create network-based features capturing relationships between 
    customers, terminals, and transaction patterns.
    """
    transactions_df = transactions_df.copy()
    
    # Set time window
    transactions_df['TX_DATETIME'] = pd.to_datetime(transactions_df['TX_DATETIME'])
    end_date = transactions_df['TX_DATETIME'].max()
    start_date = end_date - pd.Timedelta(days=window_size_days)
    
    # Group by customer
    customer_groups = transactions_df.groupby('CUSTOMER_ID')
    
    # Number of unique terminals per customer
    transactions_df['CUSTOMER_UNIQUE_TERMINALS'] = customer_groups['TERMINAL_ID'].transform('nunique')
    
    # Terminal switching speed (average time between using different terminals)
    def get_terminal_switching_speed(group):
        if len(group) <= 1:
            return 0
        group = group.sort_values('TX_DATETIME')
        terminal_changes = (group['TERMINAL_ID'] != group['TERMINAL_ID'].shift()).astype(int)
        time_diff = group['TX_DATETIME'].diff().dt.total_seconds() / 3600  # in hours
        return (terminal_changes * time_diff).mean()
    
    transactions_df['TERMINAL_SWITCHING_SPEED'] = customer_groups.apply(get_terminal_switching_speed)
    
    return transactions_df

def get_velocity_features(transactions_df, time_windows=[1, 3, 6, 12, 24]):
    """
    Create velocity features for different time windows (in hours)
    """
    features = transactions_df.copy()
    
    for window in time_windows:
        # Convert window to timedelta
        window_delta = pd.Timedelta(hours=window)
        
        # Create window start time
        features[f'window_start_{window}h'] = features['TX_DATETIME'] - window_delta
        
        # Group by customer and calculate features
        grouped = features.groupby('CUSTOMER_ID')
        
        # Number of transactions in window
        features[f'tx_count_{window}h'] = grouped.apply(
            lambda x: x.apply(
                lambda row: len(x[(x['TX_DATETIME'] > row['window_start_{window}h']) & 
                                (x['TX_DATETIME'] <= row['TX_DATETIME'])]), 
                axis=1
            )
        )
        
        # Transaction amount velocity
        features[f'amount_velocity_{window}h'] = grouped.apply(
            lambda x: x.apply(
                lambda row: x[(x['TX_DATETIME'] > row[f'window_start_{window}h']) & 
                             (x['TX_DATETIME'] <= row['TX_DATETIME'])]['TX_AMOUNT'].sum(), 
                axis=1
            )
        )
        
        # Drop temporary columns
        features = features.drop(f'window_start_{window}h', axis=1)
    
    return features

def get_time_pattern_features(transactions_df):
    """
    Create features based on temporal patterns
    """
    df = transactions_df.copy()
    
    # Extract more granular time components
    df['TX_HOUR'] = df['TX_DATETIME'].dt.hour
    df['TX_MINUTE'] = df['TX_DATETIME'].dt.minute
    df['TX_DAY_OF_WEEK'] = df['TX_DATETIME'].dt.dayofweek
    df['TX_DAY_OF_MONTH'] = df['TX_DATETIME'].dt.day
    df['TX_MONTH'] = df['TX_DATETIME'].dt.month
    
    # Create cyclical features for time components
    df['HOUR_SIN'] = np.sin(2 * np.pi * df['TX_HOUR']/24)
    df['HOUR_COS'] = np.cos(2 * np.pi * df['TX_HOUR']/24)
    df['MONTH_SIN'] = np.sin(2 * np.pi * df['TX_MONTH']/12)
    df['MONTH_COS'] = np.cos(2 * np.pi * df['TX_MONTH']/12)
    
    # Time since last transaction (per customer)
    df = df.sort_values(['CUSTOMER_ID', 'TX_DATETIME'])
    df['TIME_SINCE_LAST_TX'] = df.groupby('CUSTOMER_ID')['TX_DATETIME'].diff().dt.total_seconds()
    
    return df

def get_statistical_features(transactions_df, windows=[7, 30]):
    """
    Create features based on statistical deviations from normal patterns
    """
    df = transactions_df.copy()
    
    for window in windows:
        # Calculate rolling statistics for amount
        df[f'AMOUNT_MEAN_{window}d'] = df.groupby('CUSTOMER_ID')['TX_AMOUNT'].transform(
            lambda x: x.rolling(window, min_periods=1).mean())
        df[f'AMOUNT_STD_{window}d'] = df.groupby('CUSTOMER_ID')['TX_AMOUNT'].transform(
            lambda x: x.rolling(window, min_periods=1).std())
        
        # Z-score of current transaction amount
        df[f'AMOUNT_ZSCORE_{window}d'] = (df['TX_AMOUNT'] - df[f'AMOUNT_MEAN_{window}d']) / \
                                        df[f'AMOUNT_STD_{window}d'].replace(0, 1)
        
        # Median absolute deviation
        df[f'AMOUNT_MAD_{window}d'] = df.groupby('CUSTOMER_ID')['TX_AMOUNT'].transform(
            lambda x: x.rolling(window, min_periods=1).apply(lambda x: np.median(np.abs(x - np.median(x)))))
    
    return df

def get_ratio_features(transactions_df):
    """
    Create ratio and relationship features
    """
    df = transactions_df.copy()
    
    # Amount to average ratios
    for col in [col for col in df.columns if 'AVG_AMOUNT' in col]:
        df[f'RATIO_TO_{col}'] = df['TX_AMOUNT'] / df[col].replace(0, 1)
    
    # Transaction frequency ratios
    for col in [col for col in df.columns if 'NB_TX' in col]:
        window_size = col.split('_')[-2].replace('DAY', '')
        df[f'TX_FREQ_RATIO_{window_size}'] = df[col] / float(window_size)
    
    return df


def apply_smote_sampling(X_train, y_train, sampling_strategy=0.3, random_state=42):
    """
    Apply SMOTE sampling to the training data.    
    
    """
    # First apply mild undersampling to reduce majority class
    # This helps with memory efficiency when dealing with large datasets
    # rus = RandomUnderSampler(sampling_strategy=0.1, random_state=random_state)
    
    # Then apply SMOTE to increase minority class
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    
    # # Create a pipeline of the sampling steps
    # pipeline = Pipeline([
    #     ('undersampling', rus),
    #     ('smote', smote)
    # ])
    
    # Apply the resampling pipeline
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    return X_resampled, y_resampled
