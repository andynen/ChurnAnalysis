from sklearn.metrics import accuracy_score

def assess_model_accuracy(fitted_model, X_test, y_test):
    """
    Evaluates the accuracy of a fitted model on test data.

    Parameters:
    - fitted_model: sklearn model
        The trained model to be evaluated.
    - X_test: pd.DataFrame or np.ndarray
        Test data for features.
    - y_test: pd.Series or np.ndarray
        True labels for the test data.

    Returns:
    - accuracy: float
        Accuracy score of the model on the test data.
    """
    # Generate predictions
    y_pred = fitted_model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    return {'predictions': y_pred, 'accuracy': accuracy}

