import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm


class LithologyDataset(Dataset):
    def __init__(self, df:pd.DataFrame, labels:pd.Series, logs:list[str], num_classes:int, seq_size:int=100, interval_size:int=100, well_name_column:str='WELL') -> None:
        """
            Arguments:
            ---------
                - df (pd.DataFrame): Well log data
                - labels (pd.Series): Column containing lithology classes for each depth
                - logs (list[str]): List of logs used. Ex: GR, NPHI, ...
                - num_classes (int): Number of lithology classes
                - seq_size (int): Size of sequence sent to the model
                - interval_size (int): Size of the interval used to extract consecutive sequences
                - well_name_column (str): Name of the column that indicates the well name in the data
            Return:
            ---------
                None
        """
        
        self.data = df
        self.list_of_wells = list(df[well_name_column].unique())
        self.labels = labels
        self.logs = logs
        self.num_classes = num_classes
        self.seq_size = seq_size
        self.interval_size = interval_size
        self.well_name_column = well_name_column
        
        self.data['labels'] = labels
        self.list_of_sequences = self.__create_dataset(self.data, verbose=False)

    def __create_dataset(self, df:pd.DataFrame, verbose:bool=False) -> list:
        """
            Arguments:
            ---------
                - df (pd.DataFrame): Well log data
            Return:
            ---------
                - list_of_sequences (list): list of all sequences without null values in the dataset
        """
        
        list_of_sequences = list()
        
        if verbose:
            iterator_list_of_wells = tqdm(range(len(self.list_of_wells)))
        else:
            iterator_list_of_wells = range(len(self.list_of_wells))
            
        for i in iterator_list_of_wells:
            
            wellname = self.list_of_wells[i]
            well_df = df[df[self.well_name_column] == wellname]

            idx_null = well_df[well_df[self.logs].isnull().any(axis=1)].index.tolist()

            j=0
            while j < well_df.shape[0]-(self.seq_size-1):
                
                sequence = well_df.iloc[j:j+self.seq_size]
                
                idx_null = sequence[sequence[self.logs].isnull().any(axis=1)].index.tolist()
                
                if idx_null == []:
                    list_of_sequences.append([wellname, sequence[self.logs], sequence['labels']])
                    j = j + self.interval_size
                else:
                    j = j + idx_null[-1] + 1
                
        return list_of_sequences
    
    def __len__(self):
        
        return len(self.list_of_sequences)
    
    def __getitem__(self, idx) -> tuple[str, torch.Tensor, torch.Tensor]:
        """
            Arguments:
            ---------
                - idx (int): Index for selecting a sample from the dataset
            Return:
            ---------
                - wellname (str): Name of the well from which the sequence is taken
                - well_data_torch (torch.Tensor): Well log sequence
                - labels_torch (torch.Tensor): One-hot-encoded lithology labels sequence
        """
        
        wellname, sequence, labels = self.list_of_sequences[idx]
        # To Numpy
        sequence_numpy = sequence.to_numpy()
        sequence_numpy = np.reshape(sequence_numpy, (-1, len(self.logs)))
        
        # Create one-hot vector to represent labels
        labels_numpy = np.array([np.array([1. if i==label else 0. for i in range(self.num_classes)]) for label in labels.to_numpy()])
        
        # To Torch
        well_data_torch = torch.from_numpy(sequence_numpy).float()
        labels_torch = torch.from_numpy(labels_numpy).float()
        
        return wellname, well_data_torch, labels_torch
