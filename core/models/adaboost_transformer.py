import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from core.models.lithology_dataset import LithologyDataset


class Transformer(nn.Module):
    def __init__(self, input_dim, seq_len, num_classes, hidden_dim=2048, embed_dim=16, dropout=0.3, nhead=8):
        super(Transformer, self).__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # Simulating Embedding
        self.embed_layer = nn.Linear(input_dim, embed_dim)
        self.positional_embedding = nn.Embedding(seq_len, embed_dim)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

        # Classification Layer
        self.fc = nn.Linear(embed_dim, num_classes)

        self.eval_mode = False
        self.eval_activation = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        #x = x.permute(1, 0, 2)  # Transformer expects (S, N, C)
        batch_size, seq_len, _ = x.size()

        x = self.embed_layer(x)

        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        
        pos_embedding = self.positional_embedding(positions)
        x = x + pos_embedding
        
        x = self.transformer_encoder(x)

        x = self.fc(x)
        if self.eval_mode:
            x = self.eval_activation(x)

        x = x.view(-1, self.num_classes)

        return x


class AdaBoostTransformer:
    def __init__(self, device, logs, seq_size:int, batch_size:int=64, epochs:int=50, n_classifiers:int=50, input_dim:int=4, num_classes:int=10, hidden_dim:int=2048, embed_dim:int=64, dropout:float=0.3, learning_rate:float=0.001, verbose:bool=True):
        """
            Arguments:
            ---------
                - device (str): Device used for torch training
                - logs (list[str]): List with the name of the logs used to train models
                - seq_size (int): Sequence size
                - batch_size (int): Batch size used for training
                - epochs (int): Number of epochs trained
                - n_classifiers (int): Number of transformers trained using adaboost framework
                - input_dim (int): Size of the input dimension. Equals number of logs
                - num_classes (int): Number of classes in the data
                - hidden_dim (int): Size of the hidden dimension for each transformer
                - embed_dim (int): Size of the embeddings in the transformers
                - dropout (float): Dropout rate used in each transformer
                - learning_rate (float): Learning rate used to update model parameters
                - verbose (bool): Print state of the training process
            Return:
            ---------
                None
        """
        
        self.n_classifiers = n_classifiers
        self.logs = logs
        self.seq_size = seq_size
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.epochs = epochs
        self.device = device
        self.classifiers = [Transformer(input_dim, seq_size, num_classes, hidden_dim=hidden_dim, embed_dim=embed_dim, dropout=dropout).to(self.device) for _ in range(self.n_classifiers)]
        self.alphas = []
        self.learning_rate = learning_rate
        self.verbose = verbose

    def train_model(self, train_loader, criterion, weighted:bool=False, verbose:bool=True):
        """
        Train all transformer models with adaboost framework.
            Arguments:
            ---------
                - train_loader (torch.utils.data.DataLoader): Training well log dataloader
                - criterion (torch.nn.Module): Loss function. By default, CrossEntropyLoss
                - weighted (bool): Boolean value that indicates wether using weighted training or not
                - verbose (bool): Print state of the training process
            Return:
            ---------
                None
        """
        
        n_samples = len(train_loader.dataset) * self.seq_size
        sample_weights = torch.ones(n_samples).to(self.device) / n_samples
        
        for i, model in enumerate(tqdm(self.classifiers, disable=(not verbose))):

            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

            model.train()
            model.eval_mode = False

            # Train one transformer entirely
            for epoch in range(self.epochs):
                for batch_idx, (idxs, wellnames, well_data_torch, labels_torch) in enumerate(train_loader):
                    
                    idxs = idxs.to(self.device)
                    well_data_torch = well_data_torch.to(self.device)
                    labels_torch = labels_torch.to(self.device)

                    optimizer.zero_grad()
                    
                    outputs = model(well_data_torch)
                    
                    outputs = outputs.view(-1, self.num_classes)
                    labels_torch = labels_torch.view(-1, self.num_classes)

                    loss = criterion(outputs, labels_torch)
                    
                    if weighted:
                        
                        seq_range = torch.arange(self.seq_size, device=self.device)  # Sequence range
                        start_pos = idxs * self.seq_size
                        expanded_indices = (start_pos.unsqueeze(1) + seq_range).flatten()
                        selected_sample_weights = sample_weights[expanded_indices]
                        
                        loss = loss * (selected_sample_weights * len(sample_weights))
                    
                    loss = torch.mean(loss)

                    loss.backward()
                    optimizer.step()

            error = 0
            correct_predictions = torch.zeros(n_samples).to(self.device)
            all_targets = torch.zeros(n_samples).to(self.device)

            with torch.no_grad():
                model.eval()
                model.eval_mode = True
                # Check the missed ones by the currently trained transformer
                for idxs, wellnames, well_data_torch, labels_torch in train_loader:
                    
                    idxs = idxs.to(self.device)
                    well_data_torch = well_data_torch.to(self.device)
                    labels_torch = labels_torch.to(self.device)
                    labels_torch = labels_torch.view(-1, self.num_classes)
                    
                    outputs = model(well_data_torch)
                    predicted = torch.argmax(outputs, dim=1)
                    target = torch.argmax(labels_torch, dim=1)
    
                    correct = (predicted == target)
    
                    # Precompute start and end indices for each sample in the batch
                    batch_size = len(idxs)
                    seq_range = torch.arange(self.seq_size, device=self.device)  # Sequence range
                    start_pos = idxs * self.seq_size  # Starting positions
    
                    # Flatten and assign to correct_predictions
                    expanded_indices = (start_pos.unsqueeze(1) + seq_range).flatten()
                    correct_predictions[expanded_indices] = correct.float()
            
            error = torch.sum(sample_weights * (1 - correct_predictions)) / torch.sum(sample_weights)
            error = error.cpu().numpy()

            alpha = 0.5 * np.log((1 - error) / error)
            self.alphas.append(alpha)

            # Update sample weights
            for i in range(n_samples):
                if correct_predictions[i]:
                    sample_weights[i] *= np.exp(-alpha)
                else:
                    sample_weights[i] *= np.exp(alpha)

            sample_weights /= torch.sum(sample_weights)


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

        criterion = nn.CrossEntropyLoss(reduction='none')

        if class_weights is not None:
            self.train_model(train_dataloader, criterion, weighted=True, verbose=self.verbose)
        else:
            self.train_model(train_dataloader, criterion, weighted=False, verbose=self.verbose)
            
            
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

        y_pred_deep = []
        y_test_deep = []
        with torch.no_grad():

            y_pred_deep = np.zeros((len(dataloader.dataset)*self.seq_size, self.num_classes))

            for i, model in enumerate(tqdm(self.classifiers, disable=(not verbose))):
                model.eval()
                model.eval_mode = True
                
                alpha = self.alphas[i]
                batch_idx = 0

                for i, (idxs, wellnames, well_data_torch, labels_torch) in enumerate(dataloader):
                    
                    well_data_torch = well_data_torch.to(self.device)
                    labels_torch = labels_torch.to(self.device)
                    
                    outputs = model(well_data_torch)
                    outputs = outputs.view(-1, self.num_classes).detach().cpu().numpy()
                    
                    start_idx = batch_idx * len(outputs)
                    end_idx = start_idx + len(outputs)
                    y_pred_deep[start_idx:end_idx] += alpha * outputs
    
                    batch_idx += 1

                model.train()
                model.eval_mode = False

            y_pred_deep = np.argmax(y_pred_deep, axis=1)
                
            for i, (idxs, wellnames, well_data_torch, labels_torch) in enumerate(dataloader):
                labels_torch = labels_torch.long()
                labels_torch = labels_torch.view(-1, self.num_classes)
                test_labels = torch.argmax(labels_torch, dim=1).cpu().detach().numpy().tolist()

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

    def forward(self, x):
        """
        Give predictions for a series of sequences.
            Arguments:
            ---------
                - x (torch.Tensor): Data
            Return:
            ---------
                - y_pred_deep (list[list[int]]): Tensor of predicted classes for each row in the dataset. One hot encoded
                                                 (each row has a list of k elements, where k is the number of lithology classes). 
        """

        B, S, F = x.shape
        x = x.float()
        
        # Initialize an array to hold the weighted logits
        aggregated_logits = torch.zeros(B * S, self.num_classes, device=x.device)
        
        # Iterate through all weak classifiers and aggregate their predictions
        for i, model in enumerate(self.classifiers):
            # Forward pass through the weak classifier
            logits = model(x)  # Shape: (B, S, Num_classes)
            
            # Add the weighted logits to the aggregated logits
            aggregated_logits += self.alphas[i] * logits
        
        return None, aggregated_logits

    def eval(self):
        """
        Set all torch transformer models to evaluation mode.
            Arguments:
            ---------
                None
            Return:
            ---------
                None
        """
        for i, model in enumerate(self.classifiers):
            model.eval()
            model.eval_mode = True
