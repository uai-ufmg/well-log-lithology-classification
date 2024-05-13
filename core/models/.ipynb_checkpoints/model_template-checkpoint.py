import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import weight_norm

from abc import ABC, abstractmethod

from core.models.lithology_dataset import LithologyDataset


def train(model, dataloader, optimizer, criterion, epochs:int, num_classes:int, device:str, verbose=True) -> None:
    """
    Training pipeline for Deep Learning models.
        Arguments:
        ---------
            - model (nn.Module): Model being trained
            - dataloader (nn.DataLoader): Dataloader for loading data to the model. Either training or testing.
            - optimizer (torch.optim...): Optimizer algorithm used to update model parameters
            - criterion (nn.modules.loss...): Loss function
            - epochs (int): Number of epochs
            - num_classes (int): Number of lithology classes
            - device (str): Device to use for the training (cpu or gpu)
            - verbose (bool): Print state of the training and evaluation process
        Return:
        ---------
            None
    """
    
    for epoch in range(epochs):
        
        if verbose == True:
            print(f'Epoch {epoch+1}/{epochs}')
        
        total_loss = 0
        model.train()
        
        if verbose:
            iterator_dataloader = enumerate(tqdm(dataloader))
        else:
            iterator_dataloader = enumerate(dataloader)
        
        for i, (wellnames, well_data_torch, labels_torch) in iterator_dataloader:
            
            well_data_torch = well_data_torch.long().to(device)
            labels_torch = labels_torch.to(device)

            optimizer.zero_grad()
            
            output, probs = model(well_data_torch)
            
            # Reshape to 2D tensor (batch_size * seq_len, num_classes)
            probs = probs.view(-1, num_classes)
            
            labels_torch = labels_torch.view(-1, num_classes)
            
            loss = criterion(probs, labels_torch)
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
        
        if verbose == True:
            print(f"Training Loss: {total_loss/len(dataloader)}")


class Model(nn.Module, ABC):
    
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def forward(self, x):
        pass


    def fit(self, X_train:pd.DataFrame, y_train:pd.Series, class_weights=None) -> None:
        """
        Fit method for deep learning models.
            Arguments:
            ---------
                - X_train (pd.DataFrame): Training well log data
                - y_train (pd.Series): Training lithology class for each depth of each well
                - class_weights: Scikit learn class weight or no weight. It is used for a weighted training loss.
            Return:
            ---------
                None
        """
        
        train_dataset = LithologyDataset(X_train, y_train, self.logs, self.num_classes, seq_size=self.seq_size, interval_size=self.seq_size)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        if class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=class_weights,reduction='mean')
        else:
            criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)
        
        train(self, train_dataloader, optimizer, criterion, self.epochs, self.num_classes, self.device, verbose=self.verbose)
       
    def __evaluate(self, dataloader, num_classes:int, verbose:bool=False) -> tuple[list, list]:
        """
            Arguments:
            ---------
                - dataloader (nn.DataLoader): Dataloader for loading data to the model. Either training or testing.
                - num_classes (int): Number of lithology classes
                - verbose (bool): Print state of the training and evaluation process
            Return:
            ---------
                - y_test_deep (list[list[int]]): List of true classes for each row in the dataset. One hot encoded
                                                 (each row has a list of k elements of 0s or 1s, where k is the number of lithology classes). 
                - y_pred_deep (list[list[float]]): List of probabilities predicted for each row in the dataset. One hot encoded
                                                   (each row has a list of k elements of probabilities - floats - predicted by the model, where k is the number of lithology classes).
        """

        self.eval()
        y_pred_deep = []
        y_test_deep = []
        with torch.no_grad():
            
            if verbose:
                dataloader_iterator = enumerate(tqdm(dataloader))
            else:
                dataloader_iterator = enumerate(dataloader)
                
            for i, (wellnames, well_data_torch, labels_torch) in dataloader_iterator:

                well_data_torch = well_data_torch.long().to(self.device)
                labels_torch = labels_torch.long().to(self.device)

                output, probs = self.forward(well_data_torch)
                
                # Reshape to 2D tensor (batch_size * seq_len, num_classes)
                probs = probs.view(-1, num_classes).detach().cpu().numpy()
                
                labels_torch = labels_torch.view(-1, num_classes).detach().cpu().numpy()

                test_probs = np.argmax(probs, axis=1).tolist()
                test_labels = np.argmax(labels_torch, axis=1).tolist()

                y_pred_deep.extend(test_probs)
                y_test_deep.extend(test_labels)
            
        return y_test_deep, y_pred_deep
    
    def test(self, X_test:pd.DataFrame, y_test:pd.Series, verbose:bool=False) -> tuple[list, list]:
        """
        Test method for deep learning models. Used to evaluate the quality of models.
            Arguments:
            ---------
                - X_test (pd.DataFrame): Test well log data
                - y_test (pd.Series): Test lithology class for each depth of each well
                - verbose (bool): Print state of the training and evaluation process
            Return:
            ---------
                - y_test_deep (list[list[int]]): List of true classes for each row in the dataset. One hot encoded
                                                 (each row has a list of k elements of 0s or 1s, where k is the number of lithology classes). 
                - y_pred_deep (list[list[float]]): List of probabilities predicted for each row in the dataset. One hot encoded
                                                   (each row has a list of k elements of probabilities - floats - predicted by the model, where k is the number of lithology classes).
        """
        test_dataset = LithologyDataset(X_test, y_test, self.logs, self.num_classes, seq_size=self.seq_size, interval_size=self.seq_size)
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        y_test_deep, y_pred_deep = self.__evaluate(test_dataloader, self.num_classes, verbose)
        
        return y_test_deep, y_pred_deep
