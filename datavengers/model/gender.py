import pandas as pd
import numpy as np
import pickle as pkl

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline

from datavengers.model.predictor import Predictor

class Gender():
    def __init__(self):
      self._model = None

    def load_model(self):
        with open('./datavengers/persistence/gender/gender.model', 'rb') as fd:
            self._model = pkl.load(fd)
    
    def save_model(self):
        with open('./datavengers/persistence/gender/gender.model', 'wb') as fd:
            pkl.dump(self._model, fd) 
      
    def format_relations(self, raw_relation_data):
      relations = raw_relation_data
      relations['like_id'] = raw_relation_data['like_id'].astype(str)
      relations['like_id'] = raw_relation_data['like_id'].astype(str)
      merged_relations = relations.groupby('userid')['like_id'].apply((lambda x: "%s" % ' '.join(x))).reset_index()
      print(merged_relations.head)
      return merged_relations
    
    def train(self, raw_train_data, preTrained = 'True'):  
      if (preTrained):
        self.load_model()

      else:
        tags = raw_train_data.get_profiles()[['userid','gender']]
        tags['gender'] = tags['gender'].astype(int)
        
        merged_relations = self.format_relations(raw_train_data.get_relation())
        
        relations_data = pd.merge(merged_relations, tags, on="userid")

        X = relations_data['like_id']
        y = relations_data['gender']

        self._model = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(n_jobs=1, C=1e5, solver='liblinear')),
               ])
        
        self._model.fit(X, y)

        self.save_model()
      

    def predict(self, raw_test_data):  
      merged_relations = self.format_relations(raw_test_data.get_relation())
      X_test = merged_relations['like_id']
      return self._model.predict(X_test)