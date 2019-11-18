import pandas as pd
import numpy as np
import pickle as pkl

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

#from datavengers.model.predictor import Predictor

class Gender():
    def __init__(self):
      self._relation_model = None
      self._oxford_model = None

    def load_relation_model(self):
        with open('./datavengers/persistence/gender/relation.model', 'rb') as fd:
            self._relation_model = pkl.load(fd)
    
    def save_relation_model(self):
        with open('./datavengers/persistence/gender/relation.model', 'wb') as fd:
            pkl.dump(self._relation_model, fd) 

    def load_oxford_model(self):
        with open('./datavengers/persistence/gender/oxford.model', 'rb') as fd:
            self._oxford_model = pkl.load(fd)
    
    def save_oxford_model(self):
        with open('./datavengers/persistence/gender/oxford.model', 'wb') as fd:
            pkl.dump(self._oxford_model, fd) 

    def format_relations(self, raw_relation_data):
      relations = raw_relation_data
      relations['like_id'] = raw_relation_data['like_id'].astype(str)
      merged_relations = relations.groupby('userid')['like_id'].apply((lambda x: "%s" % ' '.join(x))).reset_index()
      return merged_relations
    
    def train(self, raw_train_data, preTrained = 'True'):  
      if (preTrained):
        self.load_relation_model()
        self.load_oxford_model()       

      else:
        # collect labels from Profile file
        tags = raw_train_data.get_profiles()[['userid','gender']]
        tags['gender'] = tags['gender'].astype(int)
        
        # train the Model for Relation data 
        merged_relations = self.format_relations(raw_train_data.get_relation())     
        relations_data = pd.merge(merged_relations, tags, on="userid")

        X_rel = relations_data['like_id']
        y_rel = relations_data['gender']

        self._relation_model = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(n_jobs=1, C=1e5, solver='liblinear')),
               ])
        
        self._relation_model.fit(X_rel, y_rel)
        self.save_relation_model()

        # train the Model for Oxford data
        oxford_train_df = raw_train_data.get_oxford()
        # Oxford file uses 'userId' instead of 'userid' ... so we apply lower case
        oxford_train_df.columns = [x.lower() for x in oxford_train_df.columns]
        oxford = pd.merge(tags, oxford_train_df, on="userid")
        oxford_facial = oxford[['userid', 'gender', 'facialhair_mustache', 'facialhair_beard', 'facialhair_sideburns']]
        oxford_facial = oxford_facial.drop_duplicates(subset='userid', keep="first")

        X_oxf = oxford_facial[['facialhair_mustache', 'facialhair_beard', 'facialhair_sideburns']]
        y_oxf = oxford_facial['gender']
        
        self._oxford_model = SVC(C=100, gamma='auto', class_weight='balanced')
        
        self._oxford_model.fit(X_oxf, y_oxf)
        self.save_oxford_model()

    def predict(self, raw_test_data):  
      relation_pred_df = self.predict_on_relation(raw_test_data)
      oxford_pred_df = self.predict_on_oxford(raw_test_data)
      # We replace 'relation predictions' by 'oxford predictions', whenever there is an oxford prediction
      # to do this we join the 2 dataframes
      relation_pred_df = relation_pred_df.set_index(['userid'])
      oxford_pred_df = oxford_pred_df.set_index(['userid'])

      relation_pref_df_copy = relation_pred_df.copy()
      relation_pred_df['gender'] = oxford_pred_df['gender']
      
      merged_df = relation_pred_df.combine_first(relation_pref_df_copy)
      
      return np.array(merged_df['gender']).astype(int)

    
    def predict_on_relation(self, raw_test_data):  
      merged_relations = self.format_relations(raw_test_data.get_relation())
      X_test = merged_relations['like_id']
      prediction = self._relation_model.predict(X_test)

      df = pd.DataFrame()
      df['userid'] = raw_test_data.get_profiles()['userid']
      df['gender'] = prediction
      return df


    def predict_on_oxford(self, raw_test_data):
      oxford_test_df = raw_test_data.get_oxford()
      oxford_test_df.columns = [x.lower() for x in oxford_test_df.columns]
      oxford_facial_df = oxford_test_df[['userid', 'facialhair_mustache', 'facialhair_beard', 'facialhair_sideburns']]
      oxford_facial_df = oxford_facial_df.drop_duplicates(subset='userid', keep="first")
      test_users = raw_test_data.get_profiles()['userid']
      oxford_data = pd.merge(test_users, oxford_facial_df, on="userid")

      X_test = oxford_data[['facialhair_mustache', 'facialhair_beard', 'facialhair_sideburns']]
      prediction = self._oxford_model.predict(X_test)

      df = pd.DataFrame()
      df['userid'] = oxford_data['userid']
      df['gender'] = prediction
      return df