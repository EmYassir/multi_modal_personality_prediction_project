from predictor import Predictor
from data import Data

import pandas as pd
import numpy as np
import pickle as pkl
from random import randint


class Gender(Predictor):

    def __init__(self):
      super().__init__()
      self._proba_table = np.array([])
      self._proba_dict = dict()
    
    def load_pretrained_model(self):
      infile = open('./datavengers/persistence/gender/gender.model','rb')
      self._probas_table = np.asarray(pickle.load(infile, encoding='bytes'))
      infile.close()
      self.set_proba_dict()
      # end of method
      
    def set_proba_dict(self):   
      for i in range(self._proba_table.shape[0]):
        uid = (self._proba_table[i][0]).astype(int)
        self._proba_dict[uid]=i  
      # end of method
    
    
    def train(self, raw_train_data, preTrained = 'True'):  
      if (preTrained):
        self.load_pretrained_model()
      else:
        gender_df = raw_train_data.get_profiles()[['userid','gender']]
        gender_df['gender'] = gender_df['gender'].astype(int)       
        relations = pd.merge(gender_df,raw_train_data.get_relation(), on="userid")
        summary = relations.groupby(['like_id','gender'],as_index=False).count()
        unique = summary['like_id'].unique()
        page_ids = list(unique)
        scores = np.zeros((len(page_ids),2))
    
        for i in range(len(page_ids)):
          page = page_ids[i]
          rows = summary.loc[ summary['like_id'] == page ]
          gender = rows.iloc[0,1]
          
          scores[i][gender] = rows.iloc[0,2]
          
          if (len(rows.index) == 2):
            gender = rows.iloc[1,1]
            scores[i][gender] = rows.iloc[1,2]
        
        scores = np.true_divide(scores, scores.sum(axis=1, keepdims=True))
        temp=np.array(page_ids)[:,np.newaxis]
        self._proba_table = np.hstack((temp,scores))
        self.set_proba_dict()
        # end of method
      
    def extract_likes(self,uid,relation):
      # returns an array with the liked pages of a given userid
      likes = []
      rels = relation.loc[ relation['userid'] == uid ]
      for i in range(len(rels)):
        likes.append(rels.iloc[i][2])
      return likes
        
    def compute_gender(self,indexes):
      #given the indexes of liked pages, returns a prediction
      # 0 for male, 1 for female
      M = 0.0
      F = 0.0     
      n = len(indexes)
      for index in indexes:   
        M += self._proba_table[index][1]/n
        F += self._proba_table[index][2]/n     
      if (M == F):
        return randint(0,1)
      else:
        return 0 if M > F else 1
      
    def occurrence_likes(self,likeids):
      # returns the indexes of occurrences in the trained data (dict)
      return [ self._proba_dict[k] for k in likeids if k in self._proba_dict ]
    
 
    def predict_user_gender(self,uid,relation):
      likes  = self.extract_likes(uid, relation)
      known_pages = self.occurrence_likes(likes)
      return self.compute_gender(known_pages)
       

    def predict(self, raw_test_data):  
      # to predict we do the following tasks:
      # 1. extract the userids from Profile
      # 2. for each userid, we:
      ###### 2.1 extract list of liked pages from Relations
      ###### 2.2 get list of corresponding indexes in trained table
      ###### 2.3 compute gender prediction
      ###### 2.4 save it in the prediction array
      # 3. return the prediction array
      
      test_profiles_df = raw_test_data.get_profiles()['userid']
      
      size = len(test_profiles_df)
      prediction = np.zeros(size)
      
      relation = raw_test_data.get_relation()
      
      for i in range(size):
        uid = test_profiles_df.iloc[i]
        prediction[i] = self.predict_user_gender(uid,relation)

      return prediction.astype(int)