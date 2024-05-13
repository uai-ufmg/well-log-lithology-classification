import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef, f1_score, precision_score, recall_score, ConfusionMatrixDisplay, balanced_accuracy_score

from .model_template import Model


class CNN(Model):
    """
    CNN: Five hidden layers, with batch norm and dropout
    
    """
    def __init__(self, cfg:dict, logs:list[str], seq_size:int, batch_size:int, lr:float, epochs:int, input_channels:int, num_classes:int, device:str, verbose:bool=True) -> None:
        
        """
            Arguments:
            ---------
                - cfg (dict): Configuration dictionary containing hyperparameters
                - logs (list[str]): List of logs used
                - seq_size (int): Size of the input sequence
                - batch_size (int): Size of the batch
                - lr (float): Learning rate for the model
                - epochs (int): Number of epochs to train the model
                - input_channels (int): Number of channels in the input data
                - num_classes (int): number of lithology classes (output size)
                - device (str): Device to use for the training (cpu or gpu)
                - verbose (bool): Print state of the training and evaluation process
            Return:
            ---------
                None
        """ 
        
        super(CNN, self).__init__()
        
        self.cfg = cfg
        self.logs = logs
        self.seq_size = seq_size
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.num_classes = num_classes
        self.device = device
        self.verbose = verbose

        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, padding='same')
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding='same')
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding='same')
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding='same')
        self.conv5 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding='same')

        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(512)

        # Dropout layers
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.2)
        self.dropout4 = nn.Dropout(0.2)
        self.dropout5 = nn.Dropout(0.2)
        
        self.relu = nn.ReLU()

        # Fully connected layer
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x:torch.Tensor):
        """
            Arguments:
            ---------
                - x (torch.Tensor): Input data for the model
            Return:
            ---------
                - x (torch.Tensor): Output data of the model
        """
        
        x = x.permute(0, 2, 1).float()   
        # Convolutional layers
        x = self.dropout1(self.relu(self.bn1(self.conv1(x))))

        x = self.dropout2(self.relu(self.bn2(self.conv2(x))))

        x = self.dropout3(self.relu(self.bn3(self.conv3(x))))

        x = self.dropout4(self.relu(self.bn4(self.conv4(x))))

        x = self.dropout5(self.relu(self.bn5(self.conv5(x))))

        # Flatten the output of the convolutional layers
        x = x.view(-1, 512)

        # Fully connected layer
        x = self.fc(x)

        return None, x
