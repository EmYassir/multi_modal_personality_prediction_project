# -*- coding: utf-8 -*-
"""
@author: Sabi

Simple example to show how to use the predictor 
"""

from datavengers.model.data import Data
from datavengers.model.predictor import Predictor
from datavengers.model.gender import Gender

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
model = Gender()

print('Using pre-trained model...')
# use False as 2nd argument to retrain...
model.train(train_data_wrapper, False)

print('Generating predictions on the test set...')
pred1 = model.predict(test_data_wrapper)
print('-> predicted:')
print(pred1)