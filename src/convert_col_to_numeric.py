import pandas as pd

def convert_col_to_numeric(df, columns):
    """
    Converts specified columns in a DataFrame to numeric data type.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    columns (list): A list of column names to be converted to numeric.
    
    Returns:
    pd.DataFrame: A DataFrame with the specified columns converted to numeric.
    """
    for col in columns:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert with coercion for invalid values
                print(f"Column '{col}' converted to numeric.")
            except Exception as e:
                print(f"Error converting column '{col}' to numeric: {e}")
        else:
            print(f"Column '{col}' does not exist in the DataFrame.")
    
    return df
