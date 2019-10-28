import numpy as np
import pandas as pd
import pickle as pkl
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from datavengers.model.predictor import Predictor


class Personality(Predictor):

    def __init__(self):
        super().__init__()
        self._layers = tuple([50,100,50])
        self._model = None
        
    
    def _rescale(self, x, oldmin, oldmax, newmin, newmax):
        coeff = float((newmax-newmin)/(oldmax - oldmin))
        return coeff * (x + oldmin) - newmin

    def _process_raw_data(self, raw_data):
        nrc = raw_data.get_nrc()
        profiles = raw_data.get_profiles()
        
        # Extracting data and combining it together
        # 1) : Ope
        yOpedf = profiles[['userid','ope']]
        yOpedf.columns = ['userId','ope']
        xOpe = pd.merge(nrc, yOpedf, on='userId', how='inner')
        
        # 2) : Ope + Con
        yCondf = profiles[['userid','con']]
        yCondf.columns = ['userId','con']
        xCon = pd.merge(xOpe, yCondf, on='userId', how='inner')
        
        # 3) : Ope + Con + Ext
        yExtdf = profiles[['userid','ext']]
        yExtdf.columns = ['userId','ext']
        xExt = pd.merge(xCon, yExtdf, on='userId', how='inner')
        
        # 4) : Ope + Con + Ext + Agr
        yAgrdf = profiles[['userid','agr']]
        yAgrdf.columns = ['userId','agr']
        xAgr = pd.merge(xExt, yAgrdf, on='userId', how='inner')
        
        # 5) : Ope + Con + Ext + Agr + Neu
        yNeudf = profiles[['userid','neu']]
        yNeudf.columns = ['userId','neu']
        
        # Merge everything and Remove Ids column
        df = (pd.merge(xAgr, yNeudf, on='userId', how='inner')).drop (columns = 'userId')
        return df
    
    def _preprocess_data(self, data_frame):
        # Transform into numpy array
        # Split features from targets
        X_np = data_frame.to_numpy()
        # rescale 'surprise' column
        X_np[:,8] = self._rescale(X_np[:,8], 0.0, 0.5, 0.0, 1.0)
        n_cls= 5
        n = X_np.shape[1]
        return X_np[:,:n - n_cls], X_np[:,n - n_cls:]
    
    def _transform_data(self, X, n = 2):
        pca = PCA(n_components = n)
        return pca.fit_transform(X)
    
    def _extract_features(self, X):
        # removing positiveness ?
        return X[:, 1:-1] 
    
    def _rescale_features(self, X):
        # rescale 'surprise'
        return X[:, 1:-1] 
    
    def _process(self, raw_data):
        df = self._process_raw_data(raw_data)
        X, y = self._preprocess_data(df)
        #X = self._transform_data(X)   # DO NOT USE NOW
        #X = self._extract_features(X) # DO NOT USE NOW
        return X, y
    
    # Public methods
    def train(self, raw_train_data):
        # Preprocess
        X, y = self._process(raw_train_data)
        # Training
        self._model = MLPRegressor(hidden_layer_sizes=self._layers)
        self._model.fit(X, y)
        
        # Train on
    
    def predict(self, raw_test_data):
        # Preprocess
        X, y = self._process(raw_test_data)
        return self._model.predict(X)
    
    def fit(self, raw_train_data):
        # Preprocess
        X, y = self._process(raw_train_data)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
        print(X_train.shape)
        print(X_test.shape)
        # Train on train data
        self._model = MLPRegressor(hidden_layer_sizes=self._layers)
        self._model.fit(X_train, y_train)
        
        # Predict on train and test sets
        y_p = self._model.predict(X_train)
        print('Train score:')
        print(self.get_rmse_score(y_p, y_train))
        y_p = self._model.predict(X_test)
        print('Test score:')
        print(self.get_rmse_score(y_p, y_test))
    
    def get_rmse_score(self, y_pred, y):
        return np.sqrt(np.mean((y_pred - y)**2, axis = 0))
    
    def update_model(self, new_model):
        self._model = new_model
    
    def load_model(self):
        with open('./datavengers/persistence/personality/personality.model', 'rb') as fd:
            self._model = pkl.load(fd)
    
    def save_model(self):
        with open('./datavengers/persistence/personality/personality.model', 'wb') as fd:
            pkl.dump(self._model, fd) 
