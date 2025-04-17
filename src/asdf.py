def prepare_data_for_modeling(df):
    """
    Prepare the data for model training.
    
    Parameters:
    df (pd.DataFrame): DataFrame with features and RUL values
    
    Returns:
    tuple: X_train, X_test, y_train, y_test, scaler, feature_names
    """
    if df is None or df.empty:
        print("No data for model training")
        return None, None, None, None, None, None
    
    # Make a copy to avoid modifying the original DataFrame
    model_df = df.copy()
    
    # Convert Timestamp to datetime
    model_df['Timestamp'] = pd.to_datetime(model_df['Timestamp'])
    
    # Extract time-based features
    model_df['Hour'] = model_df['Timestamp'].dt.hour
    model_df['DayOfWeek'] = model_df['Timestamp'].dt.dayofweek
    
    # Drop non-numeric columns
    model_df = model_df.drop(['Timestamp'], axis=1)
    
    # Define features and target
    # Exclude the target variable and any derived features that might cause data leakage
    exclude_cols = ['RUL']
    feature_cols = [col for col in model_df.columns if col not in exclude_cols]
    
    X = model_df[feature_cols]
    y = model_df['RUL']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols
