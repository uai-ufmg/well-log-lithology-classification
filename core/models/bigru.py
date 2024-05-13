import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef, f1_score, precision_score, recall_score, ConfusionMatrixDisplay, balanced_accuracy_score

from .model_template import Model


class BiGRU(Model):
    def __init__(self, cfg:dict, logs:list[str], seq_size:int, batch_size:int, lr:float, epochs:int, input_size:int, hidden_size:int, num_classes:int, device:str, num_layers:int=2, batch_first:bool=True, dropout:float=0, bidirectional:bool=True, verbose:bool=True) -> None:
        """
            Arguments:
            ---------
                - cfg (dict): Configuration dictionary containing hyperparameters
                - logs (list[str]): List of logs used
                - seq_size (int): Size of the input sequence
                - batch_size (int): Size of the batch
                - lr (float): Learning rate for the model
                - epochs (int): Number of epochs to train the model
                - input_size (int): The number of expected features in the input x (number of logs)
                - hidden_size (int): The number of features in the hidden state
                - num_classes (int): number of lithology classes (output_size)
                - device (str): Device to use for the training (cpu or gpu)
                - num_layers (int): Number of recurrent layers
                - batch_first (bool): If input is provided as (batch, seq, feature) or (seq, batch, feature)
                - dropout (float): If non-zero, introduces a Dropout layer on the outputs of each GRU layer except the last layer, with dropout probability equal to dropout.
                - bidirectional (bool): If True, becomes a bidirectional GRU.
                - verbose (bool): Print state of the training and evaluation process
                
        """ 
        super(BiGRU, self).__init__()
        
        self.logs = logs
        self.seq_size = seq_size
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.cfg = cfg
        self.device = device
        self.verbose = verbose

        self.gru = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                              batch_first=self.batch_first, bidirectional=self.bidirectional)
        
        self.fc = nn.Linear(hidden_size*2, num_classes)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
            Arguments:
            ---------
                - x (torch.Tensor): Input data for the model
            Return:
            ---------
                - x (torch.Tensor): Output tensor before softmax
                - probs (torch.Tensor): Output tensor after softmax
        """
        
        x = x.float()

        batch_size = x.size(0)

        out, hn = self.gru(x)

        x = self.fc(out)
        probs = self.softmax(x)
        
        return x, probs
