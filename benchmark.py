import os

import pandas as pd
import numpy as np
import random
from time import time

from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef, f1_score, precision_score, recall_score, ConfusionMatrixDisplay, balanced_accuracy_score

# Pytorch
import torch

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import core
from core.data import open_and_preprocess_data
from core.data.preprocessing import select_data
from core.data.scaler import Scaler
from core.models import select_model, save_model
from core.visualize_results import makeplot

from configs.config import ConfigArgs


def set_seed(seed_number:int=42, loader=None) -> np.random.RandomState:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed_number)
    np.random.seed(seed_number)
    #random_state = np.random.RandomState(seed_number)
    random.seed(seed_number)
    try:
        loader.sampler.generator.manual_seed(seed_number)
    except AttributeError:
        pass
    
    return seed_number


def predict_with_model(model, X, y=None, input_format='dl', verbose=False):
    """
        Arguments:
        ---------
            - model: Model object
            - X (pd.DataFrame): Well log data testing set
            - y (pd.Series): Lithology values for each depth
            - input_format (str): Model type. dl or ml.
            - verbose (bool): Verbose parameter.
        Return:
        ---------
            y (np.array|torch.Tensor): Ground truth label values.
            y_pred (np.array|torch.Tensor): Predicted label values.
    """
    if input_format == 'dl':
        y_test_pred, y_pred = model.test(X, y, verbose=verbose)
        return y_test_pred, y_pred
    else:
        y_pred = model.predict(X)
        return y, y_pred


def plot_predictions(model, X_plot, y_plot, le, model_name, filename, output_dir, input_format='dl', verbose=False):
    """
        Arguments:
        ---------
            - model: Model object
            - X_plot (pd.DataFrame): Well log data testing well for plots
            - y_plot (pd.Series): Lithology values for each depth in the plot well
            - le (sklearn.preprocessing.LabelEncoder): Label encoder object. Used only for XGBoost, since it requires encoding when labels are not consecutive in training
            - model_name (str): Name of the model
            - filename (str): File name used for identifying combination of parameters when saving results.
            - output_dir (str): Output directory to save plots
            - input_format (str): Model type. dl or ml.
            - verbose (bool): Verbose parameter.
        Return:
        ---------
            None
    """
    y_test_plot_pred, y_plot_pred = predict_with_model(
        model, X_plot, y_plot, input_format=input_format, verbose=verbose
    )

    y_test_inverse = le.inverse_transform(y_test_plot_pred)
    y_pred_inverse = le.inverse_transform(y_plot_pred)

    makeplot(y_test_inverse, y_pred_inverse, model_name, filename, output_dir)


def write_metrics(path_file, current_fold, y_test, y_pred, tempo_treinamento):
    """
        Arguments:
        ---------
            - path_file (str): Output directory to save metric results
            - current_fold (int): Number of current fold
            - y_test (pd.Series): Ground truth lithology values for each well in the dataset.
            - y_pred (pd.Series): Predicted lithology values for each well in the dataset.
            - tempo_treinamento (float): Amount of time elapsed from the start of the fold until the end (in seconds)
        Return:
        ---------
            None
    """
    # Choose if file will be opened on append(a) or write(w) mode.
    mode = 'a' if os.path.isfile(path_file) else 'w'
    
    with open(path_file, mode) as f:
        f.write(f'FOLD {current_fold}\n')
        f.write(f'Accuracy: {accuracy_score(y_test, y_pred)}\n')
        f.write(f'Weighted Accuracy: {balanced_accuracy_score(y_test, y_pred)}\n')
        f.write(f'MCC: {matthews_corrcoef(y_test, y_pred)}\n')
        f.write(f'Precision: {precision_score(y_test, y_pred, average="macro")}\n')
        f.write(f'Weighted Precision: {precision_score(y_test, y_pred, average="weighted")}\n')
        f.write(f'Recall: {recall_score(y_test, y_pred, average="macro")}\n')
        f.write(f'Weighted Recall: {recall_score(y_test, y_pred, average="weighted")}\n')
        f.write(f'F1-Score: {f1_score(y_test, y_pred, average="macro")}\n')
        f.write(f'Weighted F1-Score: {f1_score(y_test, y_pred, average="weighted")}\n')
        f.write(f'Training Time: {tempo_treinamento}\n\n')


def evaluate(model, model_name:str, input_format:str, X_test:pd.DataFrame, y_test:pd.Series, output_dir:str, filename:str, current_fold:int, X_test_plot:pd.DataFrame=None, y_test_plot:pd.Series=None, le:LabelEncoder|None=None, tempo_treinamento:float|None=None, verbose:bool=False, plots:bool=False) -> None:
    """
        Arguments:
        ---------
            - model: Model object
            - input_format (str): Model type. dl or ml.
            - X_test (pd.DataFrame): Well log data testing set
            - y_test (pd.Series): Lithology values for each depth
            - X_test_plot (pd.DataFrame): Well log data testing well for plots
            - y_test_plot (pd.Series): Lithology values for each depth in the plot well
            - output_dir (str): Output directory to save metric results
            - filename (str): File name used for identifying combination of parameters when saving results.
            - current_fold (int): Number of current fold
            - num_classes (int): Number of lithology classes
            - le (sklearn.preprocessing.LabelEncoder): Label encoder object. Used only for XGBoost, since it requires encoding when labels are not consecutive in training
            - tempo_treinamento (float): Amount of time elapsed from the start of the fold until the end (in seconds)
        Return:
        ---------
            None
    """
    
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f'No such file or directory: {output_dir}')

    y_test_final, y_pred_final = predict_with_model(model, X_test, y_test, input_format=input_format, verbose=verbose)

    if plots and X_test_plot is not None and y_test_plot is not None:
        plot_predictions(model, X_test_plot, y_test_plot, le, model_name, filename, output_dir,
                         input_format=input_format, verbose=verbose)

    path_file = os.path.join(output_dir, filename)
    write_metrics(path_file, current_fold, y_test_final, y_pred_final, tempo_treinamento)


def run_pipeline_kfold(cfg:dict, data:pd.DataFrame, le_dataset, train_wells, test_wells:list[str], well_names:list[str], plots:bool=True, save_trained_model:bool=False, weighted:bool=False, verbose:bool=True) -> None:
    """    
        Arguments:
        ---------
            - cfg (dict): Dictionary containing config details
            - data (pd.DataFrame): Well log data
            - le_dataset (sklearn.preprocessing.LabelEncoder): Label encoder object.
            - train_wells (tuple[list]): List with well indexes selected for training
            - test_wells (list[str]): List with well indexes selected for testing/validating
            - well_names (list[str]): List with the names of all wells
            - plots (bool): Generate and save plots showing qualitative results
            - save_trained_model (bool): Save trained model into a file
            - weighted (bool): Use weighted training to compensate for class imbalance
            - verbose (bool): Print state of the training and evaluation process
        Return:
        ---------
            None
    """
    
    scaler = Scaler(cfg['scaling_method'])
    
    for i, (train_idx, val_idx) in enumerate(train_wells):
        
        if verbose:
            print(f'Fold {i+1}')
        
        train_list = [well_names[i] for i in train_idx]
        val_list = [well_names[i] for i in val_idx]
        
        X_train, y_train = select_data(cfg, data.copy(), train_list, cfg['input_format'])
        X_val, y_val = select_data(cfg, data.copy(), val_list, cfg['input_format'])

        X_train[cfg['logs']] = scaler.fit_transform(X_train[cfg['logs']])
        X_val[cfg['logs']] = scaler.transform(X_val[cfg['logs']])
    
        model = select_model(cfg, cfg['model_name'], random_state, device, verbose=verbose)
        inicio = time()

        if cfg['input_format'] == 'dl':
            
            if weighted:
                class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
                class_weights_torch = torch.tensor(class_weights, dtype=torch.float).to(device)
                model.fit(X_train, y_train, class_weights_torch)
            else:
                model.fit(X_train, y_train)            
                
        else:

            if weighted:
                sample_weights = compute_sample_weight('balanced', y=y_train)
                model.fit(X_train.values, y_train, sample_weight=sample_weights)
            else:
                model.fit(X_train.values, y_train)
            
            X_val = X_val.values

        tempo_treinamento = time() - inicio

        plot_well = '16/10-5'
        if plots and (cfg['dataset'].lower()=='force') and (plot_well in val_list):
            val_list_plot = [plot_well]
            X_val_plot, y_val_plot = select_data(cfg, data.copy(), val_list_plot, cfg['input_format'])
            X_val_plot[cfg['logs']] = scaler.transform(X_val_plot[cfg['logs']])
            evaluate(model, cfg['model_name'], cfg['input_format'], X_val, y_val, cfg['output_dir'], cfg['filename'], i+1, X_test_plot=X_val_plot, y_test_plot=y_val_plot, le=le_dataset, tempo_treinamento=tempo_treinamento, verbose=cfg['verbose'], plots=plots)
        else:
            evaluate(model, cfg['model_name'], cfg['input_format'], X_val, y_val, cfg['output_dir'], cfg['filename'], i+1, tempo_treinamento=tempo_treinamento, verbose=cfg['verbose'])
                    
        if save_trained_model:
            save_model(model, scaler, cfg['model_name'], cfg['dataset'], cfg['seq_size'], cfg['model_dir'])


def run_pipeline(cfg:dict, random_state:int|np.random.RandomState, device:str) -> None:
    """    
        Arguments:
        ---------
            - cfg (dict): Dictionary containig config details
            - random_state (int or np.random.RandomState): Random State used for reproducibility.
            - device (str): Device in which computations will be performed (cpu or gpu)
        Return:
        ---------
            None
    """
    
    data, le_dataset, well_names, train_wells, val_wells, test_wells = open_and_preprocess_data(cfg['dataset'], cfg['data_dir'], cfg['input_format'], cfg['logs'], cfg['class_col'], 
                                                                                       cfg['split_form'], cfg['test_size'], cfg['val_size'], cfg['n_splits'], cfg['shuffle'], random_state, verbose=cfg['verbose'])
    
    # Split data based on Kfold
    run_pipeline_kfold(cfg, data, le_dataset, train_wells, test_wells, well_names, save_trained_model=cfg['save_model'], weighted=cfg['weighted'], verbose=cfg['verbose'])


if __name__=='__main__':
    
    parser = ConfigArgs()

    cfg = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random_state = set_seed(42)
        
    run_pipeline(cfg, random_state, device)
