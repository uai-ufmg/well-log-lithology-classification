import os

import json
import pandas as pd
import numpy as np
import zipfile
from urllib.parse import urljoin
import requests

from io import BytesIO

from sklearn.preprocessing import LabelEncoder

from .data import Data


class Force(Data):
    
    def __init__(self, directory:str, logs:list[str], verbose:bool=False) -> None:
        """
            Arguments:
            ---------
                - directory (str): Path to the directory where data is 
                - logs (list[str] or tuple[str]): Logs used from Petro data
                - verbose (bool): If True, print progress details. Else, does not print anything.
        """
        
        self.directory = directory
        self.logs = logs
        
        self.lithology_keys = {30000: 'Sandstone',
                     65030: 'Sandstone/Shale',
                     65000: 'Shale',
                     80000: 'Marl',
                     74000: 'Dolomite',
                     70000: 'Limestone',
                     70032: 'Chalk',
                     88000: 'Halite',
                     86000: 'Anhydrite',
                     99000: 'Tuff',
                     90000: 'Coal',
                     93000: 'Basement'}
        
        self.data, self.le = self.open_data()
    
    def standardize_names(self):
        pass
    
    def open_data(self) -> tuple[pd.DataFrame, LabelEncoder]:
        
        """
        Main method to open data.
            Arguments:
            ---------
                -
            Return:
            ---------
                - data (pd.DataFrame): Well log dataset fully configured to be used
                - le (LabelEncoder): Label Encoder used to encode lithology classes to consecutive numbers
        """
        r = requests.get(urljoin(self.directory, 'train.zip'))
        files = zipfile.ZipFile(BytesIO(r.content))

        train_data = pd.read_csv( files.open('train.csv'), sep=';' )
        hidden_test = pd.read_csv( urljoin(self.directory, 'hidden_test.csv'), sep=';' )
        leaderboard_test_features = pd.read_csv( urljoin(self.directory, 'leaderboard_test_features.csv'), sep=';' )
        leaderboard_test_target = pd.read_csv( urljoin(self.directory, 'leaderboard_test_target.csv'), sep=';' )


        ## A little of consistency checking
        leaderboard_test_target['WELL_tg'] = leaderboard_test_target.WELL
        leaderboard_test_target['DEPTH_MD_tg'] = leaderboard_test_target.DEPTH_MD
        leaderboard_test_target.drop(columns=['WELL', 'DEPTH_MD'], inplace=True)
        leaderboard_test = pd.concat([leaderboard_test_features, leaderboard_test_target], axis=1)

        ## Make sure the values for the WELL and DEPTH_MD columns match between the two concatenated data-frames
        _check_well = np.all( (leaderboard_test.WELL == leaderboard_test.WELL_tg).values )
        _check_depth = np.all( (leaderboard_test.DEPTH_MD == leaderboard_test.DEPTH_MD_tg).values )
        assert _check_well and _check_depth, 'Inconsistency found in leaderboard test data...'

        ## Passed the consistency check, we drop the redundant columns
        leaderboard_test.drop(columns=['WELL_tg', 'DEPTH_MD_tg'], inplace=True)

        ## Note leaderboard_test dataframe does not have the FORCE_2020_LITHOFACIES_CONFIDENCE column. We will therefore fill it with NaNs.
        leaderboard_test['FORCE_2020_LITHOFACIES_CONFIDENCE'] = np.nan


        data = pd.concat([train_data, leaderboard_test, hidden_test], axis=0, ignore_index=True)
        data.sort_values(by=['WELL', 'DEPTH_MD'], inplace=True)
        data.reset_index(drop=True, inplace=True)


        #data['NPHI'] = np.log(data['NPHI'])
        data['LITHOLOGY_NAMES'] = data.FORCE_2020_LITHOFACIES_LITHOLOGY.map(self.lithology_keys)
        data = data[data["LITHOLOGY_NAMES"] != 'Basement']
        le = LabelEncoder()
        data['LITHOLOGY'] = le.fit_transform(data['FORCE_2020_LITHOFACIES_LITHOLOGY'])

        return data, le
