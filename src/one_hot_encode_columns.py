import pandas as pd

def one_hot_encode_columns(df, columns_to_encode):
    """
    Converts specified variables in a DataFrame into one-hot encoding.
    
    Parameters:
    - df: pd.DataFrame
        The input DataFrame.
    - columns_to_encode: list
        List of column names to convert into one-hot encoding.

    Returns:
    - pd.DataFrame
        DataFrame with specified columns converted into one-hot encoding.
    """

    for column in columns_to_encode:
        if column in df.columns:
            # Perform one-hot encoding
            one_hot = pd.get_dummies(df[column], prefix=column)
            # Drop the original column and concatenate the one-hot encoded columns
            df = df.drop(column, axis=1).join(one_hot)
        else:
            raise ValueError(f"Column '{column}' not found in the DataFrame.")
    
    return df
