from imblearn.over_sampling import SMOTE
import pandas as pd

def apply_smote(X_train, Y_train, seed=42, k_neighbors=5):
    """
    Applies SMOTE to balance the dataset and returns a combined DataFrame.

    Parameters:
    - X_train: pd.DataFrame - Features for training.
    - Y_train: pd.Series or pd.DataFrame - Target variable for training.
    - seed: int, optional (default=42) - Random seed for SMOTE reproducibility.
    - k_neighbors: int, optional (default=5) - Number of nearest neighbors used for generating synthetic samples.

    Returns:
    - X_train_sm: pd.DataFrame - Resampled feature set.
    - Y_train_sm: pd.Series or pd.DataFrame - Resampled target variable.
    """
    # Initialize SMOTE with specified number of neighbors
    sm = SMOTE(random_state=seed, k_neighbors=k_neighbors)

    # Apply SMOTE
    X_train_sm, Y_train_sm = sm.fit_resample(X_train, Y_train)

    return X_train_sm, Y_train_sm
