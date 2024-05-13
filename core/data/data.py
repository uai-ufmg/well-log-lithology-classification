import os

import pandas as pd
import numpy as np

from abc import ABC, abstractmethod


class Data(ABC):
    
    @abstractmethod
    def open_data(self):
        pass
    
    @abstractmethod
    def standardize_names(self):
        pass
