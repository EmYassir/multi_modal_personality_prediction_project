import numpy as np
import pandas as pd
import pickle as pkl

from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import normalize

from datavengers.model.personality.data_util import Data_Util
from datavengers.model.personality.model_util import Model_Util
from datavengers.model.predictor import Predictor 





class Personality(Predictor):

    def __init__(self):
        super().__init__()
        self._liwc_data = {}
        self._targets = np.array(['ope','con','ext','agr','neu'])
        self._models = {'ope' : LinearRegression()  , 
                        'con' : Ridge(alpha=.1)  , 
                        #'ext' : SVR(kernel='rbf', C=100, gamma=1.0, epsilon=.1) , 
                        'ext' : Ridge(alpha=.1)  , 
                        'agr' : Ridge(alpha=.1)  , 
                        'neu':Ridge(alpha=.01)}
        self._selected_features = {}
        
    def _preprocess_data(self, raw_data, dic):
        liwc_df = raw_data.get_liwc()
        profiles_df = raw_data.get_profiles()
        data_util = Data_Util()
        for t in self._targets:
            dic[t] = data_util.build_df_with_target(liwc_df, profiles_df, t)
        return profiles_df.shape[0]
    
    # Public methods
    def train(self, raw_train_data):
        print('Preprocessing...')
        # Preprocess data
        self._preprocess_data(raw_train_data, self._liwc_data)
        
        # Model utility
        model_util = Model_Util()
        
        # Training
        for k, v in self._liwc_data.items():
            print('Target %s:' %k)
            print('Selecting features...')
            X = v.iloc[:,1:-1]
            y = v.iloc[:,-1]
            columns =  model_util.select_features(X, y)
            # Save selected features
            self._selected_features[k] = columns
            X_sel = X[columns]
            print('Normalizing...')
            X_normalized = normalize(X_sel, axis = 0)
    
            print('Fitting ...')
            model_util.train_model(self._models[k], X_normalized, y)
    
    def predict(self, raw_test_data):
        test_data = {}
        
        # Model utility
        model_util = Model_Util()
        # Preprocess
        size = self._preprocess_data(raw_test_data, test_data)
        result = np.empty((size,len(self._targets)))
        
        for i,t in enumerate(self._targets):
            print('Target %s:' %t)
            print('Selecting features...')
            v = test_data[t]
            X = v.iloc[:,1:-1]
            columns =  self._selected_features[t]
            X_sel = X[columns]
            print('Normalizing...')
            X_normalized = normalize(X_sel, axis = 0)
            print('Predicting...')
            model = self._models[t]
            result[:,i] = model_util.predict_from_model(model, X_normalized)
          
        return result
    
    def fit(self, raw_train_data):
        print('### FITTING FUNCTION ###')
              
        # Preprocess data
        self._preprocess_data(raw_train_data, self._liwc_data)
        
        # Loading utility
        model_util = Model_Util()
        
        # Training
        accs = {}
        for k, v in self._liwc_data.items():
            print('Target %s:' %k)
            X = v.iloc[:,1:-1]
            y = v.iloc[:,-1]
            print('Splitting data...')
            X_train, X_test, y_train, y_test = model_util.split_data(X, y, test_percent=0.2)
            columns =  model_util.select_features(X_train, y_train)
            self._selected_features[k] = columns
            print('Selecting features...')
            X_train = X_train[columns]
            X_test = X_test[columns]
            print('Normalizing...')
            X_train = normalize(X_train, axis = 0)
            X_test = normalize(X_test, axis = 0)
            print('Training...')
            model = self._models[k]
            model_util.train_model(model, X_train, y_train)
            print('Predicting...')
            acc = model_util.accuracy_model(model, X_test, y_test)
            accs[k] = acc
        return accs
            
    def fit2(self, raw_train_data):
        print('### FITTING FUNCTION ###')
             
        # Preprocess data
        self._preprocess_data(raw_train_data, self._liwc_data)
        
        # Loading utility
        model_util = Model_Util()
        
        # Training
        accs = {}
        for k, v in self._liwc_data.items():
            print('Target %s:' %k)
            X = v.iloc[:,1:-1]
            y = v.iloc[:,-1]
            
            print('Splitting data...')
            columns =  model_util.select_features(X, y)
            self._selected_features[k] = columns
            X =  X[columns] 
            X = normalize(X, axis = 0)
            X_train, X_test, y_train, y_test = model_util.split_data(X, y, test_percent=0.15)
            
            print('Training...')
            model = self._models[k]
            model_util.train_model(model, X_train, y_train)
            
            print('Predicting...')
            acc = model_util.accuracy_model(model, X_test, y_test)
            accs[k] = acc
            
        return accs
    

    def load_model(self):
        with open('./datavengers/persistence/personality/personality.model', 'rb') as fd:
            n_obj = pkl.load(fd)
            self._selected_features = n_obj._selected_features
            self._models = n_obj._models
    
    def save_model(self):
        with open('./datavengers/persistence/personality/personality.model', 'wb') as fd:
            pkl.dump(self, fd) 
