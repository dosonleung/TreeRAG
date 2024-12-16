import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from typing import Tuple, Set, Optional, List, Dict, Union

'''
each function satisfy:
input: table:dataframe, parameters
output: table:dataframe
'''

### (1) Highlight the label column 
def rename_column(table, old_col, new_col):
    table = table.rename(columns={old_col: new_col})
    print('rename_column ' + old_col + ' with ' + new_col)
    return table

### (2) Sample Label if the number is too large
def _sample_rows_by_label(table, value, n):
    # Filter rows where column 'c' equals value 'x'
    filtered_rows = table.loc[table['label'] == value]
    # Sample rows where column 'c' equals value 'x' (e.g., sampling 5 rows)
    sampled_rows = filtered_rows.sample(n=n, random_state=42)
    return sampled_rows

def sample_rows(table, value:List, sample_n:List):
    print('sample_rows ...')
    if value is None or sample_n is None:
        return table
    sample_rows = []
    for index,(v,n) in enumerate(zip(value, sample_n)):
        rows = _sample_rows_by_label(table, v, n)
        sample_rows.append(rows)
    return pd.concat(sample_rows)

### (3) Remove Unique Rows
#remove the column with values are unique=1 or unique=all
def remove_unique_columns(table, exclude_columns=[]):
    print('remove_unique_columns ...')
    vaild_cols = []
    for col in table.columns:
        if len(table[col].unique()) > 1 and len(table[col].unique()) < len(table) and not col in exclude_columns:
            vaild_cols.append(col)
    # Return a DataFrame with only non-unique columns
    return table[vaild_cols]

### (4) Imputation
### (4-1) fill mean for numerical data
def fill_with_average(table, exclude_columns=[]):
    print('fill_with_average ...')
    # Iterate over each column in the DataFrame
    for column in table.columns:
        # Check if the column is numeric
        if pd.api.types.is_numeric_dtype(table[column]) and not column in exclude_columns:
            # Calculate the mean of the column, ignoring NaN values
            mean_value = table[column].mean()
            # Fill NaN values with the mean and assign back to the column
            table[column] = table[column].fillna(mean_value)
    return table

### (4-2) encoding for category data
def fill_string_nan_with_none(table, exclude_columns=[]):
    print('fill_string_nan_with_none ...')
    # Iterate over each column in the DataFrame
    for column in table.columns:
        # Check if the column is of object type (usually string columns)
        if table[column].dtype == 'object' or table[column].dtype == 'O' and not column in exclude_columns:
            # Fill NaN values with 'None'
            table[column] = table[column].fillna('None')
    return table

def onehot_table(table, prefix_sep='_is_', exclude_columns=[]):
    # Identify string-type columns
    print('onehot_table ...')
    string_cols = table.select_dtypes(include=['object']).columns
    string_cols_ = []
    for col in string_cols:
        if exclude_columns:
            if col in exclude_columns:
                continue
        string_cols_.append(col)
    # Create one-hot encoded DataFrame for string columns
    one_hot_encoded_df = pd.get_dummies(table[string_cols_], drop_first=False, prefix_sep=prefix_sep)
    # Concatenate the one-hot encoded columns with the original DataFrame (excluding original string columns)
    df_encode = pd.concat([table.drop(columns=string_cols_), one_hot_encoded_df], axis=1)
    return df_encode

'''
example:
data_table = pipeline(
    table = data_table,
    funcs = [
        rename_column,
        sample_rows,
        remove_unique_columns,
        fill_with_average,
        fill_string_nan_with_none,
        onehot_table
    ],
    args = [
        {'old_col':'label', 'new_col':'label'},
        {'value':['normal', 'back', 'satan', 'warezclient'], 'sample_n':[1000, 1000, 1000, 1000]},
        {},
        {},
        {},
        {'exclude_columns':['label']}
    ]
)
'''
def pipeline(table, funcs:List, args:List[Dict]):
    assert len(funcs)==len(args)
    table_ = table
    for f,arg in zip(funcs, args):
        table_ = f(table_, **arg)
        print(table_.shape)
    return table_