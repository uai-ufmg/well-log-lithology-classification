import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold


def remove_quartiles(original_data:pd.DataFrame, logs:list[str], q:list=[0.01, 0.99], verbose:bool=True) -> pd.DataFrame:
    """
    Function to apply winsorization (remove outliers by clipping extreme quartiles. Upper or lower quartiles)
        Arguments:
        ---------
            - original_data (pd.DataFrame): Well log data, including lithology column
            - logs (list[str]): List of log names used. Ex: GR, NPHI, ...
            - class_col (str): Name of the lithology column
        Return:
        ---------
            - data (pd.DataFrame): Well log data without outliers.
    """
    
    data = original_data.copy()
    num_cols = len(logs)
    
    for i, col in enumerate(logs):
        if verbose:
            print(f'Handling log {i + 1}/{num_cols} - {col}')
        array_data = data[col].values
        only_nans = np.all( np.isnan(array_data) )
            
        if not only_nans:
            min_quart, max_quart = np.nanquantile(array_data, q=q)
            
            if verbose:
                print(f'{col}: min: {min_quart:.4f} - max: {max_quart:.4f} ')

            # Set outlier values as nan
            outlier_idx = np.logical_or(array_data < min_quart, array_data > max_quart)
            if verbose:
                print(f'Ignoring {np.sum(outlier_idx)} values')

            # Set series in dataframe with clipped values
            data[col] = data[col].clip(min_quart, max_quart)
            
    if verbose:
        print()
                
    return data


def removing_rows_w_missing(data:pd.DataFrame, logs:list[str], class_col:str) -> pd.DataFrame:
    """
    Function to remove missing data. Used only for ml(context-insensitive) models.
        Arguments:
        ---------
            - data (pd.DataFrame): Well log data, including lithology column
            - logs (list[str]): List of log names used. Ex: GR, NPHI, ...
            - class_col (str): Name of the lithology column
        Return:
        ---------
            - copy_data (pd.DataFrame): Well log data without rows where one of the selected logs is null.
    """
    cols_list = (logs).copy()
    cols_list.append(class_col)

    null_mask = data[cols_list].isnull().any(axis=1)
    copy_data = data[~null_mask]

    return copy_data


def data_split(data:list[str], split_form:str, test_size:float, val_size:float=None, n_splits:int=None, shuffle:bool=True, random_state:int|np.random.RandomState=42):
    """
    Data split function. Only Cross-Validation supported.
        Arguments:
        ---------
            - data (list[str]): List of well names
            - split_form (str): String indicating which form of data splitting to do
            - test_size (float): Percentage of test data. Used in train-validation-test and train-test splits.
            - val_size (float): Percentage of validation data. Only used in train-validation-test split.
            - n_splits (int): Number of splits used in KFold.
            - shuffle (bool): Shuffle data or not. Used in train-validation-test and train-test splits.
            - random_state (int or numpy.random.RandomState): Random State used for reproducibility.
        Return:
        ---------
            - train (list[str]): List of train wells.
            - val (list[str] or None): List of validation wells.
            - test (list[str] or None): List of test wells.
    """
    
    if split_form == 'kfold':
        spl_func = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        train_val = [(train_index, test_index) for train_index, test_index in spl_func.split(data)]

        return train_val, None, None

    elif split_form == 'train_val_test':
        train, test = train_test_split(data, test_size=test_size, shuffle=shuffle, random_state=random_state)
        train, val = train_test_split(train, test_size=val_size, shuffle=shuffle, random_state=random_state)

        return train, val, test

    elif split_form == 'train_test':
        train, test = train_test_split(data, test_size=test_size, shuffle=shuffle, random_state=random_state)
        return train, None, test

    else:
        raise NotImplementedError('Split method does not exist')


def select_data(cfg:dict, data_copy:pd.DataFrame, wells:list[str], input_format:str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Data selection function. Select the columns(logs).
        Arguments:
        ---------
            - cfg (dict): Dictionary containig config details
            - data_copy (pd.DataFrame): Well log data
            - wells (list): List of well names in the set - training, val, or testing.
            - input_format (str): Type of model. ml or dl.
        Return:
        ---------
            - X (pd.DataFrame): Well log data with only needed logs and without the class column.
            - y (pd.Series): Lithology Series containing all classes for each row in the dataset.
    """
    selection = data_copy[data_copy['WELL'].isin(wells)]

    if input_format == 'dl':
        X = selection[cfg['logs_info']]
    else:
        X = selection[cfg['logs']]
        
    y = selection[cfg['class_col']]

    return X, y
