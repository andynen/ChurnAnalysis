from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt

def evaluate_model_probabilities(model, X_test, Y_test):
    """
    Function to evaluate a classification model using predicted probabilities.

    Parameters:
    - model: Trained classification model with a `predict_proba` method.
    - X_test: Features of the test set.
    - Y_test: True labels of the test set.

    Outputs:
    - Prints ROC-AUC and PR-AUC scores.
    - Plots the Precision-Recall curve.
    """
    # Obtain predicted probabilities for the positive class (class 1)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Compute ROC-AUC
    roc_auc = roc_auc_score(Y_test, y_pred_proba)
    print(f"ROC-AUC Score: {roc_auc:.2f}")
    
    # Compute Precision-Recall curve and PR-AUC
    precision, recall, thresholds = precision_recall_curve(Y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    print(f"PR-AUC Score: {pr_auc:.2f}")
    
    # Plot Precision-Recall Curve
    plt.figure()
    plt.plot(recall, precision, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="upper right")
    plt.show()

# Example Usage
# evaluate_model_probabilities(churn_rf_mod_new, X_test, Y_test)
