from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle


class Scaler():
    
    def __init__(self, scaling_method='standard'):
        """
            Arguments:
            ---------
                - scaling_method (str): Scaling method used - Standard or Min-Max Scaling 

        """
        
        if scaling_method.lower() == 'standard':
            self.scaler = StandardScaler()
        elif scaling_method.lower() == 'minmax':
            self.scaler = MinMaxScaler()
        
    def fit(self, X_train):
        """
            Arguments:
            ---------
                - X_train (pd.DataFrame or np.array): Data used for fitting the scaler 
        """
        return self.scaler.fit(X_train)
        
    def fit_transform(self, X_train):
        """
            Arguments:
            ---------
                - X_train (pd.DataFrame or np.array): Data used for fitting the scaler 
        """
        return self.scaler.fit_transform(X_train)
    
    def transform(self, X):
        """
            Arguments:
            ---------
                - X (pd.DataFrame or np.array): Data used for scaling
        """
        return self.scaler.transform(X)
    
    def save(self, directory):
        """
            Arguments:
            ---------
                - directory (str): Directory/path used to save the fitted scaler
        """
        with open(directory, 'wb') as f:
            pickle.dump(self.scaler, f)
    
    def load(self, directory):
        """
            Arguments:
            ---------
                - directory (str): Directory/path from where to load the fitted scaler
        """
        with open(directory,'rb') as f:
            self.scaler = pickle.load(f)
