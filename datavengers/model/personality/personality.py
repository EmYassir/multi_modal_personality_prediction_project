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
        self._epochs['neu'] = 6
        self._epochs['ext'] = 7
        self._epochs['agr'] = 7
        self._epochs['con'] = 7
        
        # Instantiating seeds
        self._seeds={}
        self._seeds['ope'] = 42
        self._seeds['neu'] = 42
        self._seeds['ext'] = 100
        self._seeds['agr'] = 100 #42
        self._seeds['con'] = 42
        
        # Ope
        self._models['ope'] = Sequential()
        self._models['ope'].add(Dense(50, input_dim=191, kernel_initializer='normal', activation='sigmoid'))
        self._models['ope'].add(Dense(1, activation='linear'))
        self._models['ope'].summary()
        self._models['ope'].compile(loss=self._reg_util.keras_rmse, optimizer='adam', metrics=['mse'])
        
        # Neu
        self._models['neu'] = Sequential()
        self._models['neu'].add(Dense(10, input_dim=191, kernel_initializer='normal', activation='sigmoid'))
        self._models['neu'].add(Dense(1, activation='linear'))
        self._models['neu'].summary()
        self._models['neu'].compile(loss=self._reg_util.keras_rmse, optimizer='adam', metrics=['mse'])
        
        # Ext
        self._models['ext'] = Sequential()
        self._models['ext'].add(Dense(10, input_dim=191, kernel_initializer='normal', activation='sigmoid'))
        self._models['ext'].add(Dense(5, activation='relu'))  
        self._models['ext'].add(Dense(1, activation='linear'))
        self._models['ext'].summary()
        self._models['ext'].compile(loss=self._reg_util.keras_rmse, optimizer='adam', metrics=['mse'])
        
        # Agr
        self._models['agr'] = Sequential()
        self._models['agr'].add(Dense(25, input_dim=191, kernel_initializer='normal', activation='sigmoid'))
        self._models['agr'].add(Dense(1, activation='linear'))
        self._models['agr'].summary()
        self._models['agr'].compile(loss=self._reg_util.keras_rmse, optimizer='adam', metrics=['mse'])
        
        # Con
        self._models['con'] = Sequential()
        self._models['con'].add(Dense(25, input_dim=191, kernel_initializer='normal', activation='sigmoid'))
        self._models['con'].add(Dense(1, activation='linear'))
        self._models['con'].summary()
        self._models['con'].compile(loss=self._reg_util.keras_rmse, optimizer='adam', metrics=['mse'])
        
    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
    def _preprocess_data(self, raw_data):
        nrc_data = raw_data.get_nrc()
        liwc_data = raw_data.get_liwc()
        rel_data = raw_data.get_relation()
        profile_data = raw_data.get_profiles()
        # Building feature matrix
        rel = self._data_util.build_relations_df(profile_data['userid'], rel_data, 100)
        X = self._data_util.combine_nrc_liwc_rel(nrc_data, liwc_data, rel, columns_to_remove = [], transform = None)
        y = self._data_util.extract_targets(profile_data)
        return X, y
    
    # Public methods
    def train(self, raw_train_data):
        print('Train function ...')
        print('Preprocessing...')
        # Preprocess data
        X, y = self._preprocess_data(raw_train_data)
        print('Training ...')
        print('Fitting models...')
        for i, t in enumerate(self._targets):
            print('-> %s' %t)
            #self._set_seed(self._seeds[t])
            self._models[t].fit(X, y[:, i], epochs=self._epochs[t], batch_size=50,  verbose=1)
    
    def predict(self, raw_test_data):
        print('Predict function ...')
        print('Preprocessing...')
        # Preprocess data
        X, y = self._preprocess_data(raw_test_data)
        predictions = np.empty((y.shape[0],len(self._targets)))
        print('Predicting...')
        for i, t in enumerate(self._targets):
            print('-> %s' %t)
            #self._set_seed(self._seeds[t])
            y_pred = self._models[t].predict(X)
            predictions[:,i] = y_pred[:,0]
        
        return predictions
    
    def fit(self, raw_train_data):
        print('### FITTING FUNCTION (Test only) ###')
              
        # Preprocess data
        print('Preprocessing...')
        X, y = self._preprocess_data(raw_train_data)
        
        print('Splitting data...')
        X_train, X_test, y_train, y_test = self._reg_util.split_data(X, y, test_percent=0.2)
        
        # Training
        print('Fitting ...')
        history={}
        for i, t in enumerate(self._targets):
            print('-> %s' %t)
            #self._set_seed(self._seeds[t])
            history[t] = self._models[t].fit(X_train, y_train[:, i], epochs=self._epochs[t], batch_size=50,  verbose=1, validation_split=0.1)
        
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
                self._models[t] = load_model('./datavengers/persistence/personality/model_'+ str(t) +'.h5', custom_objects={'keras_rmse': self._reg_util.keras_rmse})
    
    def save_model(self):
        print('Saving models...')
        for t in self._targets:
            print('-> Saving %s model on disk ...' %t)
            self._models[t].save('./datavengers/persistence/personality/model_'+ str(t) +'.h5')  
        
'''
    def load_model(self):
        print('Loading models...')
        for t in self._targets:
            print('-> Loading %s model from disk ...' %t)
            json_file = open('./datavengers/persistence/personality/model_'+ str(t) +'.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            self._models[t] = model_from_json(loaded_model_json)
            # load weights into new model
            self._models[t].load_weights('./datavengers/persistence/personality/model_' + str(t) + '.h5')
    
    def save_model(self):
        print('Saving models...')
        for t in self._targets:
            print('-> %s:' %t)
            # Saving weights
            self._models[t].save_weights('./datavengers/persistence/personality/model_' + str(t) + '.h5')
            # Saving model
            model_json = self._models[t].to_json()
            with open('./datavengers/persistence/personality/model_'+ str(t) +'.json', 'w') as json_file:
                json_file.write(model_json)
'''

            
