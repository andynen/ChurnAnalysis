def data_characteristics( df ):
    
    """
    Summarises key characteristics of a DataFrame, including the number of rows, columns,
    column names, and data types.

    Parameters:
    ----------
    df : pandas.DataFrame - the input DataFrame for which characteristics are to be determined.

    Returns:
    -------
    str that summarises:
        - The number of rows and columns in the DataFrame.
        - A list of column names and their respective data types.
    """
    # save text in objects    
    a = f"The data has {df.shape[0]} rows and {df.shape[1]} columns"
    b = f"The column names and data types are: "
    c = f"{df.dtypes}"
    
    return f" {a} \n {b} \n {c}"