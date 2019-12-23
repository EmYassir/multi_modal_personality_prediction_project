import pandas as pd
import numpy as np
import pickle as pkl

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import QuantileTransformer, RobustScaler

from keras.models import Sequential
from keras.layers import Dense

from datavengers.model.data import Data
from datavengers.model.predictor import Predictor

class Gender(Predictor):
    """
    This class includes all of the training and prediction logic for the Gender prediction task
    """
    def __init__(self):
      # Attributes to save the trained models
      self._relation_model = None
      self._oxford_model = None
      self._nrc_liwc_model = None

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

    def load_nrc_liwc_model(self):
        with open('./datavengers/persistence/gender/nrc-liwc.model', 'rb') as fd:
            self._nrc_liwc_model = pkl.load(fd)
    
    def save_nrc_liwc_model(self):
        with open('./datavengers/persistence/gender/nrc-liwc.model', 'wb') as fd:
            pkl.dump(self._nrc_liwc_model, fd) 

    def format_relations(self, raw_relation_data):
      # Formats the relations as a concatenated string of all page likes per user
      relations = raw_relation_data
      relations['like_id'] = raw_relation_data['like_id'].astype(str)
      merged_relations = relations.groupby('userid')['like_id'].apply((lambda x: "%s" % ' '.join(x))).reset_index()
      return merged_relations


    def train(self, raw_train_data, preTrained = 'True'):
      # the training method
      if (preTrained):
        # upload the saved pre-trained models
        self.load_relation_model()
        self.load_oxford_model()
        self.load_nrc_liwc_model()

      else:
        # train the Models one by one
        # train the Model for Relation data 
        self.train_relations(raw_train_data)

        # train the Model for Oxford data
        print("training oxford")
        self.train_oxford(raw_train_data)

        # train the Model for LIWC/NRC data
        print("training NRC/LIWC")
        self.train_nrc_liwc(raw_train_data)
      return

    def train_relations(self, raw_train_data):
      # train the model with relations data
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
      # train the model with Oxford data
      tags = raw_train_data.get_profiles()[['userid','gender']]
      tags['gender'] = tags['gender'].astype(int)

      oxford_train_df = raw_train_data.get_oxford().copy()
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
        
      # We normalize the data with Robust scaler
      r_scaler = RobustScaler()
      X_oxford_scaled = r_scaler.fit_transform(X_oxford)

      # We define our NN model using Keras, one hidden layer with 100 neurons, Activation layer using sigmoid
      self._oxford_model = Sequential([
          Dense(100, activation='relu', kernel_initializer='random_normal', input_dim=X_oxford_scaled.shape[1]),
          Dense(1, activation='sigmoid'),
          ])

      self._oxford_model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
      self._oxford_model.fit(X_oxford_scaled, Y_oxford, batch_size=32, epochs=35)
      self.save_oxford_model()
      return

    def train_nrc_liwc(self, raw_train_data):
      # train the model with LIWC/NRC data (early fusion)
      tags = raw_train_data.get_profiles()[['userid','gender']]
      tags['gender'] = tags['gender'].astype(int)

      liwc_train_df = raw_train_data.get_liwc().copy()
      # liwc file uses 'userId' instead of 'userid' ... so we apply lower case
      liwc_train_df.columns = [x.lower() for x in liwc_train_df.columns]
      liwc = pd.merge(tags, liwc_train_df, on="userid")

      nrc_train_df = raw_train_data.get_nrc().copy()
      nrc_train_df.columns = [x.lower() for x in nrc_train_df.columns]

      # We merge NRC and LIWC dataframes
      nrc_liwc = pd.merge(liwc, nrc_train_df, on="userid")

      # We prepare the data for training
      Y_nrc_liwc = nrc_liwc['gender']
      X_nrc_liwc = nrc_liwc.drop(['gender','userid'], axis = 1)
        
      # We normalize the data using QuantileTransformer
      q_scaler = QuantileTransformer(100)
      X_nrc_liwc_scaled = q_scaler.fit_transform(X_nrc_liwc)

      # We define our NN model using Keras, one hidden layer with 100 neurons, Activation layer using sigmoid
      self._nrc_liwc_model = Sequential([
          Dense(100, activation='relu', kernel_initializer='random_normal', input_dim=X_nrc_liwc_scaled.shape[1]),
          Dense(1, activation='sigmoid'),
          ])

      self._nrc_liwc_model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
      self._nrc_liwc_model.fit(X_nrc_liwc_scaled, Y_nrc_liwc, batch_size=32, epochs=15)
      self.save_nrc_liwc_model()
      return

    def predict(self, raw_test_data):
      # the prediction method that implemented the hybrid fusion
      # where Oxford predictions are merged with LIWC/NRC predictions.
      # The final model does not use the predictions
      # based on Relations data

      # We first generate the uni-modal predictions
      oxford_pred_df = self.predict_on_oxford(raw_test_data)
      nrliwc_pred_df = self.predict_on_nrc_liwc(raw_test_data)

      # We replace 'liwc predictions' by 'oxford predictions', whenever there is an oxford prediction
      # to do this we join the 2 dataframes
      nrliwc_pred_df = nrliwc_pred_df.set_index(['userid'])
      oxford_pred_df = oxford_pred_df.set_index(['userid'])

      nrliwc_pref_df_copy = nrliwc_pred_df.copy()
      nrliwc_pred_df['gender'] = oxford_pred_df['gender']
      
      merged_df = nrliwc_pred_df.combine_first(nrliwc_pref_df_copy)
      
      print ("total prediction size = ", merged_df.shape)
      return np.array(merged_df['gender']).astype(int)

    
    def predict_on_relation(self, raw_test_data):
      # this method predicts gender based on relations data
      # it is not used in the final model
      merged_relations = self.format_relations(raw_test_data.get_relation())
      X_test = merged_relations['like_id']
      prediction = self._relation_model.predict(X_test)

      df = pd.DataFrame()
      df['userid'] = raw_test_data.get_profiles()['userid']
      df['gender'] = prediction
      return df


    def predict_on_oxford(self, raw_test_data):
      # This method generates predictions based on Oxford data
      oxford_test_df = raw_test_data.get_oxford().copy()
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
      r_scaler = RobustScaler()
      X_scaled = r_scaler.fit_transform(X_test)

      prediction = self._oxford_model.predict_classes(X_scaled)

      print("Oxford prediction: ", len(prediction))

      df = pd.DataFrame()
      df['userid'] = oxford_data['userid']
      df['gender'] = prediction
      return df


    def predict_on_nrc_liwc(self, raw_test_data):
      # This method generates predictions based on NRC/LIWC data
      liwc_test_df = raw_test_data.get_liwc().copy()
      nrc_test_df = raw_test_data.get_nrc().copy()
      liwc_test_df.columns = [x.lower() for x in liwc_test_df.columns]
      nrc_test_df.columns = [x.lower() for x in nrc_test_df.columns]
      test_users = raw_test_data.get_profiles()['userid']
      liwc_data = pd.merge(test_users, liwc_test_df, on="userid")

      # We make of fusion of NRC and LIWC data
      nrc_liwc_data = pd.merge(liwc_data, nrc_test_df, on="userid")

      X_test = nrc_liwc_data.drop(['userid'], axis = 1)

      # We scale the data using sklearn's QuantileTransformer
      q_scaler = QuantileTransformer(100)
      X_scaled = q_scaler.fit_transform(X_test)
      
      prediction = self._nrc_liwc_model.predict_classes(X_scaled)

      print("NRC LIWC prediction: ", len(prediction))

      df = pd.DataFrame()
      df['userid'] = nrc_liwc_data['userid']
      df['gender'] = prediction
      return df