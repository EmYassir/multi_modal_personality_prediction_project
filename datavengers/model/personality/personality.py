import numpy as np
import pandas as pd
import pickle as pkl
import copy as cp

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import normalize

from datavengers.model.personality.data_util import Data_Util
from datavengers.model.personality.model_util import Model_Util
from datavengers.model.personality.regressor_util import Regressor_Util

from datavengers.model.predictor import Predictor 







class Personality(Predictor):

    def __init__(self):
        super().__init__()
        self._targets = np.array(['ope','neu','ext','agr','con'])
        self._nrc_models = {
                'ope' : Lasso(alpha = 1.0), 
                'con' : Lasso(alpha = 1.0), 
                'ext' : Lasso(alpha = 1.0), 
                'agr' : Lasso(alpha = 1.0), 
                'neu':  Lasso(alpha = 1.0)}
        self._liwc_models = {
                'ope' : Lasso(alpha = 1.0), 
                'con' : Lasso(alpha = 1.0), 
                'ext' : Lasso(alpha = 1.0), 
                'agr' : Lasso(alpha = 1.0), 
                'neu':  Lasso(alpha = 1.0)}
        self._data_util =  Data_Util()
        self._reg_util =  Regressor_Util()
        
    def _preprocess_data(self, raw_data):
        nrc_data = raw_data.get_nrc()
        liwc_data = raw_data.get_liwc()
        profile_data = raw_data.get_profiles()
        
        X_nrc = self._data_util.get_feats(nrc_data , columns_to_remove = [], transform = 'normalize')
        X_liwc = self._data_util.get_feats(liwc_data, columns_to_remove = [], transform = 'normalize')
        y = self._data_util.extract_targets(profile_data)
        return X_nrc, X_liwc, y
    
    # Public methods
    def train(self, raw_train_data):
        print('Train function ...')
        print('Preprocessing...')
        # Preprocess data
        X_nrc, X_liwc, y = self._preprocess_data(raw_train_data)
        print('Traning with nrc data ...')
        for i, t in enumerate(self._targets):
            print('-> %s' %t)
            self._nrc_models[t].fit(X_nrc, y[:,i])
        print('Traning with liwc data ...')
        for i, t in enumerate(self._targets):
            print('-> %s' %t)
            self._liwc_models[t].fit(X_liwc, y[:,i])
    
    def predict(self, raw_test_data):
        print('Predict function ...')
        print('Preprocessing...')
        # Preprocess data
        X_nrc, X_liwc, y = self._preprocess_data(raw_test_data)
        nrc_result = np.empty((y.shape[0],len(self._targets)))
        liwc_result = np.empty((y.shape[0],len(self._targets)))
        print('Predicting with nrc...')
        for i, t in enumerate(self._targets):
            print('-> %s' %t)
            nrc_result[:,i] = self._nrc_models[t].predict(X_nrc)
        
        print('Predicting with liwc...')
        for i, t in enumerate(self._targets):
            print('-> %s' %t)
            liwc_result[:,i] = self._liwc_models[t].predict(X_liwc)
        return .5 * (nrc_result + liwc_result)
    
    def fit(self, raw_train_data):
        print('### FITTING FUNCTION (Test only) ###')
              
        # Preprocess data
        X_nrc, X_liwc, y = self._preprocess_data(raw_train_data)
        # Training
        print('Cross Validation with nrc...')
        for i, t in enumerate(self._targets):
            print('-> %s' %t)
            self._reg_util.fit_model_cv('Lasso', self._nrc_models[t], X_nrc, y[:,i], CV=10)
        
        print('Cross Validation with liwc...')
        for i, t in enumerate(self._targets):
            print('-> %s' %t)
            self._reg_util.fit_model_cv('Lasso', self._liwc_models[t], X_liwc, y[:,i], CV=10)
        

    def load_model(self):
        with open('./datavengers/persistence/personality/personality.model', 'rb') as fd:
            n_obj = pkl.load(fd)
            self._targets = n_obj._targets
            self._nrc_models =  cp.deepcopy(n_obj._nrc_models)
            self._liwc_models = cp.deepcopy(n_obj._liwc_models)
            self._data_util =  cp.deepcopy(n_obj._data_util)
            self._reg_util =  cp.deepcopy(n_obj._reg_util)
    
    def save_model(self):
        with open('./datavengers/persistence/personality/personality.model', 'wb') as fd:
            pkl.dump(self, fd) 
