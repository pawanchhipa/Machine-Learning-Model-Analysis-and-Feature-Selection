"""
Regularization and Recursive Feature Elimination implementation
"""

import pandas as pd

def make_new_feature(data: pd.DataFrame, cols: list, powers: list, new_col_name: str) -> pd.DataFrame:
    """
    Create new features by combining existing columns with powers
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe
    cols : list
        Column indices to combine
    powers : list
        Powers to raise each column to
    new_col_name : str
        Name for the new feature column
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with new feature added
    """
    try:
        data = data.copy()  # Don't modify original
        data[new_col_name] = 1
        for power, col in zip(powers, cols):
            data[new_col_name] = data[new_col_name] * (data[cols[col]] ** power)
        return data
    except Exception as e:
        raise ValueError(f"Error creating new feature: {str(e)}")