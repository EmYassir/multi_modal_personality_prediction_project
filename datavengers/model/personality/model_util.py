import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import copy 


class Model_Util:
    
    def __init__(self):
        pass
   
    def select_features(self, X, y, verbose = False):
        fsel = LassoCV(cv=5, max_iter=1000000, tol = 0.1)
        fsel.fit(X, y)
        if verbose:
            print("Best alpha using built-in LassoCV: %f" %fsel.alpha_)
            print("Best score using built-in LassoCV: %f" %fsel.score(X,y))
            
        coef = pd.Series(fsel.coef_, index = X.columns)
        if verbose:
            print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated "
                  + "the other " +  str(sum(coef == 0)) + " variables")
        selected_features = np.array(coef[coef != 0].axes[0])
        return selected_features
    '''
    def select_features(self, X, y, verbose = False):
        return np.array(X.columns)
     '''
    def split_data(self, X, y, test_percent=0.15):
        return train_test_split(X, y, test_size=test_percent)
        
    def train_model(self, model, X_train, y_train):
        model.fit(X_train, y_train)
        return model
    
    def predict_from_model(self, model, X_test):
        return model.predict(X_test)
    
    def accuracy_model(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        # RMSE
        return np.sqrt(np.mean((y_pred - y_test)**2, axis = 0))
        

class StackingAveragedModels:
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
   
    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = copy.deepcopy(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = copy.deepcopy(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
                
        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
    #Do the predictions of all base models on the test data and use the averaged predictions as 
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)
    
    def score(self, y_pred, y_test):
        # RMSE
        return np.sqrt(np.mean((y_pred - y_test)**2, axis = 0))