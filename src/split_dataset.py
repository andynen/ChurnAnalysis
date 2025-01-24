from sklearn.model_selection import train_test_split

def split_dataset(df, target_column, test_size=0.2, random_state=None):
    """
    Splits a dataset into training and test data.

    Parameters:
    - df: pd.DataFrame
        The full dataset containing features and the target variable.
    - target_column: str
        The name of the target column in the DataFrame.
    - test_size: float, optional (default=0.2)
        Proportion of the dataset to include in the test split (between 0.0 and 1.0).
    - random_state: int, optional (default=None)
        Seed used by the random number generator for reproducibility.

    Returns:
    - X_train: Training data for features.
    - X_test: Test data for features.
    - Y_train: Training data for target variable.
    - Y_test: Test data for target variable.
    """
    # Separate features and target
    X = df.drop(columns=[target_column])
    Y = df[target_column]
    
    # Split the dataset into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, Y_train, Y_test
