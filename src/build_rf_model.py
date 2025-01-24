from sklearn.ensemble import RandomForestClassifier

def build_rf_model(X_train, y_train, n_estimators=500, oob_score=True, n_jobs=4, 
                            random_state=50, max_features="sqrt", max_leaf_nodes=30):
    """
    Fit a Random Forest model.

    Parameters:
    - X_train: pd.DataFrame or np.ndarray
        Training data for features.
    - y_train: pd.Series or np.ndarray
        Training data for the target variable.
    - n_estimators: int, optional (default=500)
        The number of trees in the forest.
    - oob_score: bool, optional (default=True)
        Whether to use out-of-bag samples to estimate the generalization accuracy.
    - n_jobs: int, optional (default=4)
        The number of jobs to run in parallel. -1 means using all processors.
    - random_state: int, optional (default=50)
        Seed used by the random number generator for reproducibility.
    - max_features: str or int, optional (default="auto")
        The number of features to consider when looking for the best split.
    - max_leaf_nodes: int, optional (default=30)
        Maximum number of leaf nodes in each tree.

    Returns:
    - fitted_model: RandomForestClassifier - The fitted Random Forest model.
    """
    # Create the Random Forest model
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        oob_score=oob_score,
        n_jobs=n_jobs,
        random_state=random_state,
        max_features=max_features,
        max_leaf_nodes=max_leaf_nodes
    )
    
    # Fit the model to the training data
    rf_model.fit(X_train, y_train)
    
    return rf_model
