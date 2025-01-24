import pandas as pd
from sklearn.preprocessing import LabelEncoder

def label_encode_columns(df, columns_to_encode):
    """
    Converts specified categorical variables in a DataFrame to label-encoded values.
    
    Parameters:
    - df: pd.DataFrame
        The input DataFrame.
    - columns_to_encode: list
        List of column names to label encode.

    Returns:
    - pd.DataFrame
        DataFrame with specified columns label-encoded.
    """
    
    # Store LabelEncoders for reference if needed
    label_encoders = {} 
    
    for column in columns_to_encode:
        if column in df.columns:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            label_encoders[column] = le    # Save the label encoder for reversibility or reuse. 
        else:
            raise ValueError(f"Column '{column}' not found in the DataFrame.")
    
    return df
