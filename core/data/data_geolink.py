import os

import lasio
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import zipfile

import gdown

from sklearn.preprocessing import LabelEncoder

from .data import Data


class Geolink(Data):
    
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
        
        self.lithology_keys = {
                    35:'Aeolian Sandstone',
                    22:'Anhydrite',
                    12:'Argillaceous Limestone',
                    36:'Arkose',
                    23:'Basement',
                    25:'Biogenic Ooze',
                    16:'Calcareous Cement',
                    31:'Calcareous Debris Flow',
                    14:'Calcareous Shale',
                    33:'Carnallite',
                    9:'Chalk',
                    19:'Cinerite',
                    18:'Coal',
                    17:'Conglomerate',
                    3:'Cross Bedded Sst',
                    15:'Dolomite',
                    26:'Gap',
                    21:'Halite',
                    34:'Kainite',
                    11:'Limestone',
                    13:'Marlstone',
                    30:'Metamorphic Rock',
                    24:'Plutonic Rock',
                    32:'Polyhalite',
                    10:'Porous Limestone',
                    1:'Sandstone',
                    4:'Sandy Silt',
                    8:'Shale',
                    6:'Shaly Silt',
                    5:'Silt',
                    2:'Silty Sand',
                    7:'Silty Shale',
                    29:'Spiculite',
                    27:'Sylvinite',
                    28:'Volcanic Rock',
                    20:'Volcanic Tuff'
                }
        
        self.std_names = {'DEPT': 'DEPTH_MD'}
        
        self.data, self.le = self.open_data(verbose=verbose)
    
    def standardize_names(self, df:pd.DataFrame) -> pd.DataFrame:
        '''
        Change column names in the dataset to match the standard names.
            Arguments:
            ---------
                - df (pd.DataFrame): Well log dataset
            Return:
            ---------
                - df (pd.DataFrame): Well log dataset with standardized log names
        '''
        
        df = df.rename(columns=self.std_names)
        return df
    
    def open_data(self, verbose:bool) -> tuple[pd.DataFrame, LabelEncoder]:
        
        """
        Main method to open data.
            Arguments:
            ---------
                - verbose (bool): If True, print progress details. Else, does not print anything.
            Return:
            ---------
                - data (pd.DataFrame): Well log dataset fully configured to be used
                - le (LabelEncoder): Label Encoder used to encode lithology classes to consecutive numbers
        """
        dataset_file = os.path.join('datasets', 'geolink.zip')
        print(os.getcwd())

        if not os.path.exists('datasets'):
            os.mkdir('datasets')
            gdown.download(f'{self.directory}1ZvLG1SRBQB4mDUmPHSc_6kP-ubS5uNCv', dataset_file, quiet=False)
            with zipfile.ZipFile(dataset_file, 'r') as zip_ref:
                zip_ref.extractall(os.path.join('datasets', 'geolink'))
        else:
            if not os.path.isfile(dataset_file):
                gdown.download(f'{self.directory}1ZvLG1SRBQB4mDUmPHSc_6kP-ubS5uNCv', dataset_file, quiet=False)
                with zipfile.ZipFile(dataset_file, 'r') as zip_ref:
                    zip_ref.extractall(os.path.join('datasets', 'geolink'))

        self.directory = os.path.join('datasets', 'geolink')

        list_wells_files = os.listdir(self.directory)

        f = os.path.join(self.directory, list_wells_files[0])
        las = lasio.read(f)
        df_wells_geolink = las.df()
        df_wells_geolink = df_wells_geolink.reset_index()
        df_wells_geolink["WELL"] = list_wells_files[0][:-4]

        # iterate over files in that directory
        if verbose:
            iterator_files = tqdm(list_wells_files[1:])
        else:
            iterator_files = list_wells_files[1:]
            
        for filename in iterator_files:
            f = os.path.join(self.directory, filename)
            # checking if it is a file
            if (os.path.isfile(f) and '.las' in f):
                las = lasio.read(f)
                df_i = las.df()
                df_i = df_i.reset_index()
                df_i["WELL"] = filename[:-4]
                df_wells_geolink = pd.concat([df_wells_geolink, df_i])
                
        df_wells_geolink = self.standardize_names(df_wells_geolink)

        df_wells_geolink["LITHOLOGY_NUMBERS"] = df_wells_geolink["LITHOLOGY_GEOLINK"]
        df_wells_geolink["LITHOLOGY_GEOLINK"] = df_wells_geolink["LITHOLOGY_GEOLINK"].map(self.lithology_keys)
        le = LabelEncoder()
        df_wells_geolink['LITHOLOGY'] = le.fit_transform(df_wells_geolink['LITHOLOGY_GEOLINK'])

        return df_wells_geolink, le
