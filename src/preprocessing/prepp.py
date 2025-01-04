import pandas as pd  
import datetime  
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE 
import warnings
warnings.filterwarnings('ignore')

def get_train_test_set(transactions_df,
                       start_date_training,
                       delta_train=7, delta_delay=7, delta_test=7,
                       sampling_ratio=1,
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
    
    # Apply SMOTE to balance the training data
    print("Applying SMOTE to the training data...")
    smote = SMOTE(sampling_strategy=sampling_ratio, random_state=random_state)
    datetime_columns = ['TX_DATETIME']  # Exclude datetime columns for SMOTE
    X_train = train_df.drop(columns=['TX_FRAUD'] + datetime_columns)
    y_train = train_df['TX_FRAUD']
    
    # Resample the training data
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    train_df_resampled = pd.concat([X_train_res, y_train_res], axis=1)
    
    # Re-add datetime columns and sort datasets
    train_df_resampled = pd.concat([train_df_resampled, train_df[datetime_columns].reset_index(drop=True)], axis=1)
    train_df_resampled['TX_DATETIME'] = train_df_resampled['TX_DATETIME'].fillna(method='ffill')
    train_df_resampled = train_df_resampled.sort_values('TRANSACTION_ID')
    test_df = test_df.sort_values('TRANSACTION_ID')
    
    return train_df_resampled, test_df

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