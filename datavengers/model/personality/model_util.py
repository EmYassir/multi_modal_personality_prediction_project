import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split

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
        
