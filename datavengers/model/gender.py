import pandas as pd
import numpy as np
import pickle as pkl

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense

from datavengers.model.data import Data
from datavengers.model.predictor import Predictor

class Gender(Predictor):
    def __init__(self):
      self._relation_model = None
      self._oxford_model = None
      self._liwc_model = None

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

    def load_liwc_model(self):
        with open('./datavengers/persistence/gender/liwc.model', 'rb') as fd:
            self._liwc_model = pkl.load(fd)
    
    def save_liwc_model(self):
        with open('./datavengers/persistence/gender/liwc.model', 'wb') as fd:
            pkl.dump(self._liwc_model, fd) 

    def format_relations(self, raw_relation_data):
      relations = raw_relation_data
      relations['like_id'] = raw_relation_data['like_id'].astype(str)
      merged_relations = relations.groupby('userid')['like_id'].apply((lambda x: "%s" % ' '.join(x))).reset_index()
      return merged_relations


    def train(self, raw_train_data, preTrained = 'True'):
      if (preTrained):
        self.load_relation_model()
        self.load_oxford_model()
        self.load_liwc_model()

      else:
        # train the Model for Relation data 
        self.train_relations(raw_train_data)

        # train the Model for Oxford data
        self.train_oxford(raw_train_data)

        # train the Model for LIWC data
        self.train_liwc(raw_train_data)
      return

    def train_relations(self, raw_train_data):
      tags = raw_train_data.get_profiles()[['userid','gender']]
      tags['gender'] = tags['gender'].astype(int)

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
      return

    def train_oxford(self, raw_train_data):
      tags = raw_train_data.get_profiles()[['userid','gender']]
      tags['gender'] = tags['gender'].astype(int)

      oxford_train_df = raw_train_data.get_oxford()
      # Oxford file uses 'userId' instead of 'userid' ... so we apply lower case
      oxford_train_df.columns = [x.lower() for x in oxford_train_df.columns]
      oxford = pd.merge(tags, oxford_train_df, on="userid")
        
      # We add a column with face size
      oxford['face_size'] = oxford['facerectangle_width'] * oxford['facerectangle_height']

      # We only keep the face with biggest size 
      oxford_unique_face = oxford.sort_values('face_size').drop_duplicates(['userid'], keep='last')
        
      # We prepare the data for training
      Y_oxford = oxford_unique_face['gender']
      X_oxford = oxford_unique_face.drop(['userid','face_size','faceid','gender'], axis = 1)
        
      # We normalize the data
      min_max_scaler = MinMaxScaler()
      X_oxford_scaled = min_max_scaler.fit_transform(X_oxford)

      # We define our NN model using Keras, one hidden layer with 100 neurons, Activation layer using sigmoid
      self._oxford_model = Sequential([
          Dense(100, activation='relu', kernel_initializer='random_normal', input_dim=X_oxford_scaled.shape[1]),
          Dense(1, activation='sigmoid'),
          ])

      self._oxford_model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
      self._oxford_model.fit(X_oxford_scaled, Y_oxford, batch_size=32, epochs=30)
      self.save_oxford_model()
      return

    def train_liwc(self, raw_train_data):
      tags = raw_train_data.get_profiles()[['userid','gender']]
      tags['gender'] = tags['gender'].astype(int)

      liwc_train_df = raw_train_data.get_liwc()
      liwc_train_df.columns = [x.lower() for x in liwc_train_df.columns]
      liwc = pd.merge(tags, liwc_train_df, on="userid")

      # We prepare the data for training
      Y_liwc = liwc['gender']
      X_liwc = liwc.drop(['gender','userid'], axis = 1)
        
      # We normalize the data
      min_max_scaler = MinMaxScaler()
      X_liwc_scaled = min_max_scaler.fit_transform(X_liwc)

      # We define our NN model using Keras, one hidden layer with 100 neurons
      self._liwc_model = Sequential([
          Dense(100, activation='relu', kernel_initializer='random_normal', input_dim=X_liwc_scaled.shape[1]),
          Dense(1, activation='sigmoid'),
          ])

      self._liwc_model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
      self._liwc_model.fit(X_liwc_scaled, Y_liwc, batch_size=32, epochs=30)
      self.save_liwc_model()
      return


    def predict(self, raw_test_data):
      oxford_pred_df = self.predict_on_oxford(raw_test_data)
      liwc_pred_df = self.predict_on_liwc(raw_test_data)

      # We replace 'liwc predictions' by 'oxford predictions', whenever there is an oxford prediction
      # to do this we join the 2 dataframes
      liwc_pred_df = liwc_pred_df.set_index(['userid'])
      oxford_pred_df = oxford_pred_df.set_index(['userid'])

      liwc_pref_df_copy = liwc_pred_df.copy()
      liwc_pred_df['gender'] = oxford_pred_df['gender']
      
      # the dataframe below contains the merge of oxford predictions that replace liwc predictions 
      merged_df = liwc_pred_df.combine_first(liwc_pref_df_copy)
      
      return np.array(merged_df['gender']).astype(int)


    def predict_on_relation(self, raw_test_data):
      # Not used in the submission of Dec 2nd, as we are testing LIWC this time
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
      # We add a column with face size
      oxford_test_df['face_size'] = oxford_test_df['facerectangle_width'] * oxford_test_df['facerectangle_height']

      # We only keep the face with biggest size 
      oxford_test_unique_face = oxford_test_df.sort_values('face_size').drop_duplicates(['userid'], keep='last')

      test_users = raw_test_data.get_profiles()['userid']
      oxford_data = pd.merge(test_users, oxford_test_unique_face, on="userid")

      # We remove non-required fields
      X_test = oxford_data.drop(['userid','face_size','faceid'], axis = 1)

      # We normalize the data
      min_max_scaler = MinMaxScaler()
      X_scaled = min_max_scaler.fit_transform(X_test)

      # We use Keras predict_classes to get the class label
      prediction = self._oxford_model.predict_classes(X_scaled)

      df = pd.DataFrame()
      df['userid'] = oxford_data['userid']
      df['gender'] = prediction
      return df


    def predict_on_liwc(self, raw_test_data):
      liwc_test_df = raw_test_data.get_liwc()
      liwc_test_df.columns = [x.lower() for x in liwc_test_df.columns]
      test_users = raw_test_data.get_profiles()['userid']
      liwc_data = pd.merge(test_users, liwc_test_df, on="userid")

      # We remove non-required fields
      X_test = liwc_data.drop(['userid'], axis = 1)

      # We normalize the data
      min_max_scaler = MinMaxScaler()
      X_scaled = min_max_scaler.fit_transform(X_test)

      # We use Keras predict_classes to get the class label
      prediction = self._liwc_model.predict_classes(X_scaled)

      df = pd.DataFrame()
      df['userid'] = liwc_data['userid']
      df['gender'] = prediction
      return df
