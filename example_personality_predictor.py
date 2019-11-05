# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 13:08:49 2019

@author: Yassir


Simple example to show how to use the predictor 
"""
import numpy as np
from datavengers.model.data import Data
from datavengers.model.predictor import Predictor
from datavengers.model.personality.personality import Personality


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
    
print('Instantiating model')
predictor = Personality()
print('Fitting model...') # Method only used for testing!!!
accs = predictor.fit(train_data_wrapper)
print('Printing results:') 
targets = np.array(['ope','neu','ext','agr','con'])
for k in targets:
    print('\'%s\' : %.3f ' %(k, round(accs[k] * 100.0, 3)))
    

print('Re-training model on the whole set...')
predictor.train(train_data_wrapper)
print('Serializing model...')
predictor.save_model()
print('Generating predictions on the test set...')
pred1 = predictor.predict(test_data_wrapper)
print('-> predicted:')
print(pred1)

print('Loading model: Use of pre-trained model...')
predictor.load_model()
print('Generating predictions on the test set...')
pred2 = predictor.predict(test_data_wrapper)
print('-> predicted:')
print(pred2)
print('-> difference:')
print(np.sum(pred1 - pred2, axis = 0))
