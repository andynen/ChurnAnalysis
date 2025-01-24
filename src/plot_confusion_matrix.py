import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix( y_true, y_pred, title = "Confusion Matrix", figsize = (4, 3) ):
    """
    Plots the confusion matrix as a heatmap.

    Parameters:
    - y_true: array-like - Ground truth (true labels).
    - y_pred: array-like - Predicted labels.
    - title: str, optional (default="Confusion Matrix") - Title of the plot.
    - figsize: tuple, optional (default=(4, 3)) - Size of the figure.
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot the heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt="d", linecolor="k", linewidths=3, cmap="Blues")
    plt.title(title, fontsize=14)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
