import numpy as np
import pandas as pd
import pickle as pkl
import random



from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
#from tensorflow.python.keras.models import Sequential
#from tensorflow.python.keras.layers import Dense
#from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
import tensorflow as tf

from datavengers.model.personality.data_util import Data_Util
from datavengers.model.personality.model_util import Model_Util
from datavengers.model.personality.regressor_util import Regressor_Util

from datavengers.model.predictor import Predictor 
from keras.models import model_from_json
from keras.models import load_model

from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform



class Personality(Predictor):

    def __init__(self):
        super().__init__()
        self._targets = np.array(['ope','neu','ext','agr','con'])
        self._data_util =  Data_Util()
        self._reg_util =  Regressor_Util()
        # Instantiating models
        self._models={}
        # Instantiating epochs
        self._epochs={}
        self._epochs['ope'] = 4
        self._epochs['neu'] = 7
        self._epochs['ext'] = 7
        self._epochs['agr'] = 7
        self._epochs['con'] = 7
        
        # Instantiating seeds
        self._seeds={}
        self._seeds['ope'] = 42
        self._seeds['neu'] = 12
        self._seeds['ext'] = 42
        self._seeds['agr'] = 42 
        self._seeds['con'] = 42
        
        
    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
    def _preprocess_data(self, raw_data):
        nrc = raw_data.get_nrc()
        liwc = raw_data.get_liwc()
        profile = raw_data.get_profiles()
        print('Processing profile dataframe...')
        profile_df = self._data_util.preprocess_profile_df(profile)
        print('profile_df columns',  profile_df.columns)
        reference = profile_df['userId']
        print('Gathering features...')
        print('nrc columns',  nrc.columns)
        print('liwc columns',  liwc.columns)
        nrc_df = self._data_util.align_features_df(nrc, reference)
        liwc_df = self._data_util.align_features_df(liwc, reference)
        print('Combining features...')
        feats_df = self._data_util.combine_features([nrc_df, liwc_df], reference)
        print('Extracting targets...')
        targets_df = self._data_util.extract_targets_df(profile_df, reference)
        print('Getting pre-processed data set...')
        X = self._data_util.extract_data(feats_df)
        y = self._data_util.extract_data(targets_df)
        return X, y
    
    
    def _init_models(self, dimensions):
        # Ope
        self._models['ope'] = Sequential()
        self._models['ope'].add(Dense(50, input_dim=dimensions, kernel_initializer='normal', activation='sigmoid'))
        self._models['ope'].add(Dense(1, activation=self._reg_util.custom_activation))
        #self._models['ope'].summary()
        self._models['ope'].compile(loss=self._reg_util.keras_rmse, optimizer='adam', metrics=['mse'])
        
        # Neu
        self._models['neu'] = Sequential()
        self._models['neu'].add(Dense(25, input_dim=dimensions, kernel_initializer='normal', activation='sigmoid'))
        self._models['neu'].add(Dense(1, activation=self._reg_util.custom_activation))
        #self._models['neu'].summary()
        self._models['neu'].compile(loss=self._reg_util.keras_rmse, optimizer='adam', metrics=['mse'])
        
        # Ext
        self._models['ext'] = Sequential()
        self._models['ext'].add(Dense(50, input_dim=dimensions, kernel_initializer='normal', activation='sigmoid'))
        self._models['ext'].add(Dense(1, activation=self._reg_util.custom_activation))
        #self._models['ext'].summary()
        self._models['ext'].compile(loss=self._reg_util.keras_rmse, optimizer='adam', metrics=['mse'])
        
        # Agr
        self._models['agr'] = Sequential()
        self._models['agr'].add(Dense(25, input_dim=dimensions, kernel_initializer='normal', activation='sigmoid'))
        self._models['agr'].add(Dense(8, activation='relu')) 
        self._models['agr'].add(Dense(1, activation=self._reg_util.custom_activation))
        #self._models['agr'].summary()
        self._models['agr'].compile(loss=self._reg_util.keras_rmse, optimizer='adam', metrics=['mse'])
        
        # Con
        self._models['con'] = Sequential()
        self._models['con'].add(Dense(50, input_dim=dimensions, kernel_initializer='normal', activation='sigmoid'))
        self._models['con'].add(Dense(1, activation=self._reg_util.custom_activation))
        #self._models['con'].summary()
        self._models['con'].compile(loss=self._reg_util.keras_rmse, optimizer='adam', metrics=['mse'])
        
    # Public methods
    def train(self, raw_train_data):
        print('Train function ...')
        print('Preprocessing...')
        # Preprocess data
        X, y = self._preprocess_data(raw_train_data)
        print('Initializing models...')
        self._init_models(X.shape[1])
        print('Training ...')
        print('Fitting models...')
        for i, t in enumerate(self._targets):
            print('-> %s' %t)
            self._set_seed(self._seeds[t])
            self._models[t].fit(X, y[:, i], epochs=self._epochs[t], batch_size=100,  verbose=False)
    
    def predict(self, raw_test_data):
        print('Predict function ...')
        print('Preprocessing...')
        # Preprocess data
        X, y = self._preprocess_data(raw_test_data)
        predictions = np.empty((y.shape[0],len(self._targets)))
        print('Predicting...')
        for i, t in enumerate(self._targets):
            print('-> %s' %t)
            y_pred = self._models[t].predict(X)
            predictions[:,i] = y_pred[:,0]
        
        return predictions
    
    def fit(self, raw_train_data):
        print('### FITTING FUNCTION (Test only) ###')
        # Preprocess data
        print('Preprocessing...')
        X, y = self._preprocess_data(raw_train_data)
        print('Initializing models...')
        self._init_models(X.shape[1])
        print('Splitting data...')
        X_train, X_test, y_train, y_test = self._reg_util.split_data(X, y, test_percent=0.2)
        
        # Training
        print('Fitting ...')
        history={}
        for i, t in enumerate(self._targets):
            print('-> %s' %t)
            self._set_seed(self._seeds[t])
            history[t] = self._models[t].fit(X_train, y_train[:, i], epochs=self._epochs[t], batch_size=100,  verbose=False, validation_split=0.1)
        
        print('Predicting...')
        for i, t in enumerate(self._targets):
          #self._set_seed(self._seeds[t])
          y_pred = self._models[t].predict(X_test)
          print('-> %s: %f %%' %(t, self._reg_util.score(y_pred[:,0], y_test[:, i])))
          
    def load_model(self):
        print('Loading models...')
        for t in self._targets:
            print('-> Loading %s model from disk ...' %t)
            with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
                self._models[t] = load_model('./datavengers/persistence/personality/model_'+ str(t) +'.h5', custom_objects={'keras_rmse': self._reg_util.keras_rmse,
                            'custom_activation': self._reg_util.custom_activation})
    
    def save_model(self):
        print('Saving models...')
        for t in self._targets:
            print('-> Saving %s model on disk ...' %t)
            self._models[t].save('./datavengers/persistence/personality/model_'+ str(t) +'.h5')  
  
