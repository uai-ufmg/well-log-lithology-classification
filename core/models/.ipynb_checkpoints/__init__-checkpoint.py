"""
 Import necessary modules and functions for lithology prediction.
 BiLSTM, BiGRU, and CNN modules for different neural network architectures.
 GenericDataset module for handling generic datasets.
 Train function for training the models.
"""
import pandas as pd
import numpy as np
from tqdm import tqdm

# Scikit Learn
from sklearn.svm import SVC # SVM
from sklearn.neighbors import KNeighborsClassifier # KNN
from sklearn import tree # Decision Tree
from sklearn.ensemble import RandomForestClassifier # Random Forest
from sklearn.neural_network import MLPClassifier # Multi-layer perceptron(MLP)
from sklearn.naive_bayes import GaussianNB # Naive-Bayes

# XGBoost
from xgboost import XGBClassifier

import torch
import torch.nn as nn

from .bilstm import BiLSTM
from .bigru import BiGRU
from .cnn import CNN


def select_model(cfg:dict, model_name:str, random_state:int|np.random.RandomState, device:str, verbose:bool=True):
    """
    Function for selecting which model to use.
        Arguments:
        ---------
            - cfg (dict): Configuration dictionary containing hyperparameters
            - model_name (str): Name of the model to be selected
            - random_state (int or numpy.random.RandomState): Random State used for reproducibility.
            - device (str): Device to use for the training (cpu or gpu)
            - verbose (bool): Print state of the training and evaluation process
        Return:
        ---------
            - model: Model object selected. Can be of any model type implemented or imported.
    """
    
    model_name = model_name.lower()
    
    if model_name == 'xgb':
        model = XGBClassifier(
            n_estimators = cfg['n_estimators'],
            max_depth = cfg['max_depth'],
            max_leaves = cfg['max_leaves'],
            grow_policy = cfg['grow_policy'],
            booster = cfg['booster'],
            objective = cfg['objective'],
            eval_metric = cfg['eval_metric'],
            eta = cfg['eta'],
            min_child_weight = cfg['min_child_weight'],
            max_delta_step = cfg['max_delta_step'],
            subsample = cfg['subsample'],
            colsample_bytree = cfg['colsample_bytree'],
            gamma = cfg['gamma'],
            reg_alpha = cfg['reg_alpha'],
            reg_lambda = cfg['reg_lambda'],
            verbosity = 1,
            random_state = random_state
        )
    
    elif model_name == 'rf':
        model = RandomForestClassifier(
            n_estimators = cfg['n_estimators'],
            criterion = cfg['criterion'],
            min_samples_split = cfg['min_samples_split'],
            min_samples_leaf = cfg['min_samples_leaf'],
            max_leaf_nodes = cfg['max_leaf_nodes'],
            max_features = cfg['max_features'],
            max_depth = cfg['max_depth'],
            bootstrap = cfg['bootstrap'],
            random_state = random_state,
            class_weight = cfg['class_weight'],
            max_samples = cfg['max_samples']
        )
    
    elif model_name == 'nb':
        model = GaussianNB(
            priors = cfg['priors'],
            var_smoothing = cfg['var_smoothing']
        )
    
    elif model_name == 'mlp':
        model = MLPClassifier(
            hidden_layer_sizes = cfg['hidden_layer_sizes'], # The ith element represents the number of neurons in the ith hidden layer.
            activation = cfg['activation'],
            solver = cfg['solver'],
            alpha = cfg['alpha'],
            batch_size = cfg['batch_size'],
            learning_rate = cfg['learning_rate'],
            learning_rate_init = cfg['learning_rate_init'],
            power_t = cfg['power_t'],
            max_iter = cfg['max_iter'],
            shuffle = cfg['shuffle'],
            random_state = random_state,
            tol = cfg['tol'],
            momentum = cfg['momentum'],
            nesterovs_momentum = cfg['nesterovs_momentum'],
            early_stopping = cfg['early_stopping'],
            validation_fraction = cfg['validation_fraction'],
            beta_1 = cfg['beta_1'],
            beta_2 = cfg['beta_2'],
            epsilon = cfg['epsilon'],
            n_iter_no_change = cfg['n_iter_no_change'],
            max_fun = cfg['max_fun']
        )
    
    elif model_name == 'bilstm':
        
        model = BiLSTM(cfg = cfg,
                       logs = cfg['logs'],
                       seq_size = cfg['seq_size'],
                       batch_size = cfg['batch_size'],
                       lr = cfg['lr'],
                       epochs = cfg['epochs'],
                       input_size = len(cfg['logs']),
                       hidden_size = cfg['hidden_size'],
                       num_classes = cfg['num_classes'],
                       device = device,
                       num_layers=cfg['num_layers'],
                       verbose=verbose).to(device)
        
    elif model_name == 'bigru':
        
        model = BiGRU(cfg = cfg,
                      logs = cfg['logs'],
                      seq_size = cfg['seq_size'],
                      batch_size = cfg['batch_size'],
                      lr = cfg['lr'],
                      epochs = cfg['epochs'],
                      input_size = len(cfg['logs']),
                      hidden_size = cfg['hidden_size'],
                      num_classes = cfg['num_classes'],
                      device = device,
                      num_layers = cfg['num_layers'],
                      verbose = verbose).to(device)
        
    elif model_name == 'cnn':
        
        model = CNN(cfg = cfg,
                    logs = cfg['logs'],
                    seq_size = cfg['seq_size'],
                    batch_size = cfg['batch_size'],
                    lr = cfg['lr'],
                    epochs = cfg['epochs'],
                    input_channels = cfg['num_logs'],
                    num_classes = cfg['num_classes'],
                    device = device,
                    verbose = verbose).to(device)
    
    return model


def save_model(model, scaler, model_name:str, dataset_name:str, seq_size:int, directory:str) -> None:
    
    """
    Save trained model in a desired path.
        Arguments:
        ---------
            - model: Model object. Can be of any model type implemented or imported.
            - scaler (core.data.scaler.Scaler): Scaler object.
            - model_name (str): Name of the model.
            - dataset_name (str): Name of the dataset.
            - seq_size (int): Sequence size used in data preprocessing.
            - directory (str): Directory in which model is saved.
        Return:
        ---------
            None
    """
    
    model_name = model_name.lower()
    
    scaler.save(os.path.join(directory, f"scaler_{model_name}_{dataset_name}.pkl"))
    
    if model_name == 'xgb':
        model.save_model(os.path.join(directory, f"xgb_{dataset_name}.json"))

    elif model_name == 'rf':
        joblib.dump(model, os.path.join(directory, f"rf_{dataset_name}.joblib"))
    
    elif model_name == 'nb':
        with open(os.path.join(directory, f'nb_{dataset_name}.pkl'),'wb') as f:
            pickle.dump(model,f)
    
    elif model_name == 'mlp':
        with open(os.path.join(directory, f'mlp_{dataset_name}.pkl'),'wb') as f:
            pickle.dump(model,f)
    
    elif model_name == 'bilstm':
        
        torch.save(model.state_dict(), os.path.join(directory, f'bilstm_{dataset_name}_{seq_size}.pt'))
        
    elif model_name == 'bigru':
        
        torch.save(model.state_dict(), os.path.join(directory, f'bigru_{dataset_name}_{seq_size}.pt'))
        
    elif model_name == 'cnn':
        
        torch.save(model.state_dict(), os.path.join(directory, f'cnn_{dataset_name}_{seq_size}.pt'))
