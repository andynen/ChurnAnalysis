import pandas as pd

def convert_to_binary(df, columns_to_binary):
    """
    Converts variables with two labels into binary format in the given DataFrame.
    
    Parameters:
    - df: pd.DataFrame
        The input DataFrame.
    - columns_to_binary: list
        List of column names to convert into binary format.

    Returns:
    - pd.DataFrame
        DataFrame with specified columns converted into binary format.
    """
    
    for column in columns_to_binary:
        if column in df.columns:
            unique_values = df[column].unique()
            if len(unique_values) == 2:
                # Map the two unique values to 0 and 1
                mapping = {unique_values[0]: 0, unique_values[1]: 1}
                df[column] = df[column].map(mapping)
            else:
                raise ValueError(f"Column '{column}' does not have exactly two unique values.")
        else:
            raise ValueError(f"Column '{column}' not found in the DataFrame.")
    
    return df
