import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def automated_eda_plotly(data, target_var):
    """
    Automatically performs EDA for categorical and numerical variables against
    the target variable using Plotly for interactive visualizations.

    Parameters:
    ----------
    data : pd.DataFrame
        The dataset to analyze.
    target_var : str
        The target variable column name (e.g., 'churn').

    Returns:
    -------
    None
        Displays interactive plots for each variable.
    """
    # Ensure target variable is in the dataset
    if target_var not in data.columns:
        raise ValueError(f"Target variable '{target_var}' not found in the DataFrame.")

    # Pie chart for the target variable
    print("Generating pie chart for the churn distribution...")
    pie_chart = px.pie(
        data, names=target_var, 
        title=f"Distribution of {target_var}", 
        hole=0.3,
        width=500, height=400  # Set plot size
    )
    pie_chart.show()

    # Separate numerical and categorical variables
    numerical_cols = data.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

    # Remove the target variable from these lists
    if target_var in numerical_cols:
        numerical_cols.remove(target_var)
    if target_var in categorical_cols:
        categorical_cols.remove(target_var)

    print(f"Numerical columns: {numerical_cols}")
    print(f"Categorical columns: {categorical_cols}")

    # Graphical analysis for numerical variables
    print("Generating plots for numerical variables...")
    for col in numerical_cols:
        # Boxplot
        fig = px.box(
            data, x=target_var, y=col, color=target_var, 
            title=f"Boxplot of {col} by {target_var}", 
            template="plotly",
            width=600, height=400  # Set plot size
        )
        fig.show()

    # Graphical analysis for categorical variables
    print("Generating plots for categorical variables...")
    for col in categorical_cols:
        # Countplot
        fig = px.histogram(
            data, x=col, color=target_var, barmode="group",
            title=f"Countplot of {col} by {target_var}", 
            template="plotly",
            width=600, height=400  # Set plot size
        )
        fig.update_xaxes(title_text=col)
        fig.update_yaxes(title_text="Count")
        fig.show()
