# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 13:08:49 2019

@author: Yassir

Simple example to show how to use the predictor 
"""

from datavengers.model.data import Data
from datavengers.model.predictor import Predictor
from datavengers.model.personality import Personality


liwc_train = './data/Train/Text/liwc.csv'
nrc_train = './data/Train/Text/nrc.csv'
rel_train = './data/Train/Relation/Relation.csv'
oxf_train = './data/Train/Image/oxford.csv'
pro_train = './data/Train/Profile/Profile.csv'
    
#### Test data
liwc_test = './data/Public_Test/Text/liwc.csv'
nrc_test = './data/Public_Test/Text/nrc.csv'
rel_test = './data/Public_Test/Relation/Relation.csv'
oxf_test = './data/Public_Test/Image/oxford.csv'
pro_test = './data/Public_Test/Profile/Profile.csv'

#### Creating data containers
print('Preparing data...')
train_data_wrapper = Data(liwc_train, nrc_train, rel_train, oxf_train, pro_train)
test_data_wrapper = Data(liwc_test, nrc_test, rel_test, oxf_test, pro_test)
    
print('Instantiating model: Calling with no arguments (\'\')...')
model = Personality()
print('Fitting model...')
model.fit(train_data_wrapper)
print('Re-training model on the whole set...')
model.train(train_data_wrapper)
print('Generating predictions on the test set...')
pred1 = model.predict(test_data_wrapper)
print('-> predicted:')
print(pred1)
print('Serializing model...')
model.save_model()
print('Loading model: Calling with \'use_predefined_model\' = True...')
model = Personality(True)
print('Generating predictions on the test set...')
pred2 = model.predict(test_data_wrapper)
print('-> predicted:')
print(pred2)
