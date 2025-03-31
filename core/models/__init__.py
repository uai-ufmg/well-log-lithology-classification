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
from .resnet import ResNet18, ResNet50, ResNet101, ResNet152
from .adaboost_transformer import AdaBoostTransformer
from .hnfcl import HNFCL


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

    match model_name:
        case "xgb":
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
    
        case 'rf':
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
    
        case 'nb':
            model = GaussianNB(
                priors = cfg['priors'],
                var_smoothing = cfg['var_smoothing']
            )
    
        case 'mlp':
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
    
        case 'bilstm':
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
            
        case 'bigru':
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
        
        case 'cnn':
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

        case 'resnet':
            model = ResNet50(cfg = cfg,
                        logs = cfg['logs'],
                        seq_size = cfg['seq_size'],
                        batch_size = cfg['batch_size'],
                        lr = cfg['lr'],
                        epochs = cfg['epochs'],
                        channels = cfg['num_logs'],
                        num_classes = cfg['num_classes'],
                        device = device,
                        verbose = verbose).to(device)

        case 'adaboost_transformer':
            model = AdaBoostTransformer(n_classifiers = cfg['n_classifiers'],
                        hidden_dim = cfg['hidden_dim'],
                        input_dim = len(cfg['logs']),
                        logs = cfg['logs'],
                        seq_size = cfg['seq_size'],
                        batch_size = cfg['batch_size'],
                        learning_rate = cfg['lr'],
                        dropout = cfg['dropout'],
                        epochs = cfg['epochs'],
                        num_classes = cfg['num_classes'],
                        device = device,
                        verbose = verbose)

        case 'hnfcl':
            cfg_if = {'n_estimators': cfg['n_estimators_if'], 'max_features': cfg['max_features_if']}
            cfg_gbdt = {'n_estimators': cfg['n_estimators_gbdt'], 'max_depth': cfg['max_depth_gbdt'], 'loss': cfg['loss_gbdt'], 'learning_rate': cfg['learning_rate_gbdt']}
            cfg_rf = {'n_estimators': cfg['n_estimators_rf'], 'max_depth': cfg['max_depth_rf'], 'criterion': cfg['criterion_rf']}
            cfg_et = {'n_estimators': cfg['n_estimators_et'], 'max_depth': cfg['max_depth_et'], 'criterion': cfg['criterion_et']}
            
            model = HNFCL(cfg_if = cfg_if,
                          cfg_gbdt = cfg_gbdt,
                          cfg_rf = cfg_rf,
                          cfg_et = cfg_et,
                          random_state = random_state,
                          verbose = verbose)

        case match_model:
            raise NotImplementedError(f"Model name '{match_model}' does not match any implemented model!")
            
    
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
    
    match model_name:
        case "xgb":
            model.save_model(os.path.join(directory, f"xgb_{dataset_name}.json"))

        case 'rf':
            joblib.dump(model, os.path.join(directory, f"rf_{dataset_name}.joblib"))
    
        case 'nb':
            with open(os.path.join(directory, f'nb_{dataset_name}.pkl'),'wb') as f:
                pickle.dump(model,f)
    
        case 'mlp':
            with open(os.path.join(directory, f'mlp_{dataset_name}.pkl'),'wb') as f:
                pickle.dump(model,f)
    
        case 'bilstm':
            torch.save(model.state_dict(), os.path.join(directory, f'bilstm_{dataset_name}_{seq_size}.pt'))
        
        case 'bigru':
            torch.save(model.state_dict(), os.path.join(directory, f'bigru_{dataset_name}_{seq_size}.pt'))
        
        case 'cnn':
            torch.save(model.state_dict(), os.path.join(directory, f'cnn_{dataset_name}_{seq_size}.pt'))

        case 'resnet':
            torch.save(model.state_dict(), os.path.join(directory, f'resnet_{dataset_name}_{seq_size}.pt'))

        case 'adaboost_transformer':
            dict_state_values = dict()
            for i, transformer in enumerate(model.classifiers):
                dict_state_values[f'transformer{i}_state_dict'] = transformer.state_dict()
                
            torch.save(dict_state_values, os.path.join(directory, f'transformer_{dataset_name}_{seq_size}.pt'))

        case 'hnfcl':
            model.save(directory, f'hnfcl_{dataset_name}')

        case match_model:
            raise NotImplementedError(f"Model name '{match_model}' does not match any implemented model!")
