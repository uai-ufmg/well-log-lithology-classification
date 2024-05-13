# Import open_data function for loading datasets.
# Import removing_rows_w_missing, data_split, and select_data functions for preprocessing data.
# Import Scaler class for scaling data.

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

#from Frente2_Benchmark_Litologia.core.data.open_datasets import open_data
from .data_force import Force
from .data_geolink import Geolink
from .preprocessing import removing_rows_w_missing, data_split, select_data, remove_quartiles
from .scaler import Scaler


def open_data(dataset_name:str, data_dir:str, logs:list[str], verbose:bool) -> pd.DataFrame:
    
    """
    Function that opens data according to the dataset wanted.
        Arguments:
        ---------
            - dataset_name (str): Name of the dataset (Force, Geolink, Petro, ...)
            - data_dir (str): Path for folder containing dataset
            - logs (str): List of names of logs used.
            - verbose (bool): If True, print progress details. Else, does not print anything.
        Return:
        ---------
            - data (pd.DataFrame): Well log dataset
            - le (LabelEncoder): Label Encoder used to encode lithology classes to consecutive numbers
    """
    
    dataset_name = (dataset_name).lower()
    
    if dataset_name == 'force':
        force_dataset = Force(data_dir, logs)
        data, le = force_dataset.data, force_dataset.le
    elif dataset_name == 'geolink':
        geolink_dataset = Geolink(data_dir, logs)
        data, le = geolink_dataset.data, geolink_dataset.le
    else:
        raise NotImplementedError('Dataset name not supported')
        
    return data, le


def open_and_preprocess_data(dataset_name:str, data_dir:str, model_type:str, logs:list[str], class_col:str, split_form:str, test_size:float, val_size:float, n_splits:int, shuffle:bool, random_state:int|np.random.RandomState, verbose:bool=True) -> tuple[pd.DataFrame, LabelEncoder, list, list, list]:
    
    """
    Function that receives all necessary parameters to open and preprocess data and calls all necessary functions, classes and methods.
        Arguments:
        ---------
            - dataset_name (str): Name of the dataset (Force, Geolink, Petro, ...)
            - data_dir (str): Path for folder containing dataset 
            - model_type (str): ml or dl. ml means machine learning and it is used for context-insensitive models. dl means deep learning and 
                                it is used for context-sensitive, i.e., models that receive sequences as input.
            - logs (str): List of names of logs used.
            - class_col (str): Name of the label column (usually 'Lithology')
            - split_form (str): Splitting strategy - Cross Validation(others not supported yet)
            - test_size (float): Size of test set. Range: 0-1.
            - val_size (float): Size of validation set. Range: 0-1.
            - n_splits (int): Number of splits used in Cross Validation.
            - shuffle (bool): Wether to shuffle or not while data splitting.
            - random_state (int or np.random.RandomState): Random state to define random operations.
            - verbose (bool): If True, print progress details. Else, does not print anything.
        Return:
        ---------
            - data (pd.DataFrame): Well log dataset fully configured to be used
            - le (LabelEncoder): Label Encoder used to encode lithology classes to consecutive numbers
            - well_names (list[str]): List of all well names contained in dataset
            - train_wells (list[str]): List of train wells after splitting
            - val_wells (list[str] or None): List of validation wells after splitting. Can be None if there is no validation split.
            - test_wells (list[str] or None): List of test wells after splitting. Can be None if there is no test split.
    """
    
    model_type = model_type.lower()
    
    data, le = open_data(dataset_name, data_dir, logs, verbose=verbose)
    
    data = remove_quartiles(data, logs, verbose=verbose)
    
    if model_type.lower() == 'ml':
        data = removing_rows_w_missing(data, logs, class_col)
    
    well_names = list(data['WELL'].unique())
    train_wells, val_wells, test_wells = data_split(data=well_names, 
                                                   split_form=split_form,
                                                   test_size=test_size,
                                                   val_size=val_size,
                                                   n_splits=n_splits,
                                                   shuffle=shuffle,
                                                   random_state=random_state)
    
    return data, le, well_names, train_wells, val_wells, test_wells
