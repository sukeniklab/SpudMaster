import os
import pandas as pd
import numpy as np
from typing import Tuple, List

#Just to read and align dataframes for calculations to be performed
possible_file_types = ['csv', 'pkl']

def get_dataframes_in_dir(path, file_type=possible_file_types):
    tmp =1

def read_dataframe_from_directory(df_path, file_type):
    tmp =1

def get_column_index(columns: List[str], column_name: str) -> int:
    """
    Return the index of a column in a list of columns.
    Raises a ValueError if the column is not found.
    """
    if column_name not in columns:
        raise ValueError(f"Column '{column_name}' not found in column list.")
    return columns.index(column_name)

def remove_columns_to_left(df: pd.DataFrame, upto: str) -> pd.DataFrame:
    """
    Removes all columns to the left of the specified column (including all prior ones).
    """
    df = df.copy()
    column_list = df.columns.tolist()
    column_index = get_column_index(column_list, upto)
    return df.iloc[:, column_index:]

def columns_len_different(column_1: List[str], column_2: List[str]) -> bool:
    """
    Returns True if the lengths of the two column lists are different.
    """
    return len(column_1) != len(column_2)
    
def is_not_in_list(value, value_list)-> bool:
    """
    Returns True if value is not in the list.
    """
    return value not in value_list

def get_unique_columns(column_1: List[str], column_2: List[str]) -> List[str]:
    """
    Returns a list of column names that are not common between the two lists.
    """
    return np.setxor1d(np.array(column_1), np.array(column_2)).tolist()

def set_column_types(df: pd.DataFrame, column_list: List[str], data_type= str) -> pd.DataFrame:
    """
    Sets the data type of each column in column_list to the specified data_type.
    """
    for column in column_list:
        if column not in df.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame.")
        df[column] = df[column].astype(data_type)
    return df

def split_df_values(df: pd.DataFrame, column: str='construct', values_to_left: list | str = 'all') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits a DataFrame into two based on values in a specified column.
    
    Parameters:
    - column: column to split by
    - values_to_left: values to keep in the left DataFrame. Use 'all' to return full df and None
    
    Returns:
    - left_value_df: DataFrame with matching values
    - right_value_df: DataFrame with excluded values
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame.")    
    
    if values_to_left == 'all':
        return df.copy(), None

    # Convert string to list if needed
    if isinstance(values_to_left, str):
        values_to_left = [values_to_left]

    # Ensure all requested values are in the DataFrame
    values_to_left = [val for val in values_to_left if val in df[column].unique()]

    left_value_df = df[df[column].isin(values_to_left)].copy()
    right_value_df = df[~df[column].isin(values_to_left)].copy()

    return left_value_df, right_value_df 

def set_intermediate_column(df : pd.DataFrame, column_name: list[str], position: list[int]) -> pd.DataFrame:
    """
    Adds new columns with default value 0 at specified positions in the DataFrame.
    """
    if len(column_names) != len(positions):
        raise ValueError("Length of column_names and positions must match.")
    
    
    for index, column in enumerate(column_list):
        val = 0
        
        df[column] = val
        col = df.pop(column)
        
        pos = position[index]
        df = df.insert(val, column, col)

    return df

def get_opposite(column_set, current_col):
    """
    Given a list of two column sets, return the one that is not current_col.
    """
    return column_set[1] if current_col == column_set[0] else column_set[0]

def get_columns_list(column_1, column_2) -> (list, list): 
    """
    Returns two lists of column names that are present in one list but not the other.
    """
    
    column_set = [columns_1, columns_2]
    unique_list = get_unique_columns(columns_1, columns_2)

    if len(unique_list) != 0:
        list_dic = {}
        for index, set_col in enumerate(column_set):
            opposite_column_list = get_opposite(column_set, set_col)
            list_dic[index] = [x for x in unique_list if x not in opposite_column_list]
        return list_dic[0], list_dic[1]
    else:
        return [], []

def set_new_columns(df_act, df_col_ref, columns_needed):
    """
    Adds missing columns to df_act based on reference column order in df_col_ref.
    """
    index_list = [get_column_index(item, df_col_ref) for item in columns_needed]
    return set_intermediate_column(df_act, columns_needed, index_list)  

def add_columns_to_dataframe(df1, df2) -> (pd.DataFrame, pd.DataFrame): 
    """
    Ensures df1 and df2 have the same columns by adding missing ones with default values (0),
    inserted at appropriate positions to match the column order of the other.
    """
    df1_columns = df1.columns.tolist()
    df2_columns = df2.columns.tolist()

    df1_col_needed, df2_col_needed = get_columns_list(df1_columns, df2_columns)
    if df1_col_needed:
        df1 = set_new_columns(df1, df2_columns, df1_col_needed)
    if df2_col_needed:
        df2 = set_new_columns(df2, df1_columns, df2_col_needed)
    return df1, df2
        
