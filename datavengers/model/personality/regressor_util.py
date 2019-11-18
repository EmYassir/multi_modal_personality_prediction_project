import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from mlxtend.regressor import StackingCVRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


class Regressor_Util:
    
    def __init__(self):
        self._pca = None
        pass
    
    
    def set_n_componens(self, n):
        self._pca = PCA(n)
    
    def set_feat_selector(self, model):
        self._feat_selector = model
    
    def fit_feat_selector(self, X, y):
        self._feat_selector.fit(X, y)
    
    def feat_selector_transform(self, X):
        return self._feat_selector.transform(X)
        
    def fit_pca_transformer(self, X):
        self._pca.fit(X)
    
    def pca_transform_features(self, X):
        return self._pca.transform(X)
    
    def split_data(self, X, y, test_percent=0.20):
        return train_test_split(X, y, test_size=test_percent)
        
    def train_model(self, model, X_train, y_train):
        model.fit(X_train, y_train)
        
    def rescale_data(self, X, lower, upper):
        return MinMaxScaler(feature_range=[lower, upper]).fit_transform(X)
        
    def score_model(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        # RMSE
        return np.sqrt(np.mean((y_pred - y_test)**2, axis = 0))
    
    def score(self, y_pred, y_test):
        # RMSE
        return np.sqrt(np.mean((y_pred - y_test)**2, axis = 0))
        
    def test_multiple_models(self, models, X_train, X_test, y_train, y_test):
        accs = {}
        for k, v in models.items():
            print('Training with model %s ...' %k)
            model = models[k]
            self.train_model(model, X_train, y_train)
            acc = self.score_model(model, X_test, y_test)
            print('RMSE = %s' %acc)
            accs[k] = acc
        return accs
    
    def fit_model_cv(self, model_name, model, X, y, CV=5):
        print('Fitting model %s' %model_name)
        kfold = KFold(n_splits=CV, shuffle=True, random_state=156)
        
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0],))
        scores = []
        for train_index, test_index in kfold.split(X, y):
            model.fit(X[train_index], y[train_index])
            y_pred = model.predict(X[test_index])
            scores.append(self.score(y_pred, y[test_index]))
            out_of_fold_predictions[test_index] = y_pred
        scores=np.array(scores)
        print('RMSE = %f, std = %f' %(scores.mean(),scores.std()))
        return out_of_fold_predictions, scores
    
    def fit_stacked_models_cv(self, model_1, model_2, meta_model, X_1, X_2, y, CV=5):
        print('Fitting stacked models')
        kfold = KFold(n_splits=CV, shuffle=True, random_state=156)
        
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        print('training model_1')
        oof_preds_1 = np.zeros((X_1.shape[0],))
        for train_index, test_index in kfold.split(X_1, y):
            model_1.fit(X_1[train_index], y[train_index])
            y_pred = model_1.predict(X_1[test_index])
            oof_preds_1[test_index] = y_pred
        
        print('training model_2')
        oof_preds_2 = np.zeros((X_2.shape[0],))
        for train_index, test_index in kfold.split(X_2, y):
            model_2.fit(X_2[train_index], y[train_index])
            y_pred = model_2.predict(X_2[test_index])
            oof_preds_2[test_index] = y_pred
        
        print('Combining data...')
        
        X = np.column_stack((oof_preds_1, oof_preds_2))
        oof_preds = np.zeros((X.shape[0],))
        print('training stacked model')
        scores = []
        for train_index, test_index in kfold.split(X, y):
            meta_model.fit(X[train_index], y[train_index])
            y_pred = meta_model.predict(X[test_index])
            scores.append(self.score(y_pred, y[test_index]))
            oof_preds[test_index] = y_pred
        scores=np.array(scores)
        print('RMSE = %f, std = %f' %(scores.mean(),scores.std()))
        return oof_preds, scores
    
    
    def test_model(self, model_name, model, X_train, X_test, y_train, y_test):
        print('Training with model %s ...' %model_name)
        self.train_model(model, X_train, y_train)
        acc = self.score_model(model, X_test, y_test)
        print('RMSE = %s' %acc)
        return acc
    
    def test_model_CV(self, model_name, model, X, y, CV = 5):
        
        scores = cross_val_score(model, X, y, scoring = 'r2', cv=CV)
        print(scores)
        sqrt_scores = np.sqrt(scores)
        rmse = np.mean(sqrt_scores)
        std = np.std(sqrt_scores)
        print("R^2 Score: %f (+/- %f) [%s]" % (rmse, std, model_name))
        return rmse
    
    '''
    def select_features(self, X, y, verbose = False):
        reg = LassoCV(cv=5, max_iter=1000000, tol = 0.1)
        reg.fit(X, y)
        if verbose:
            print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
            print("Best score using built-in LassoCV: %f" %reg.score(X,y))
        coef = pd.Series(reg.coef_, index = X.columns)
        if verbose:
            print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated "
                  + "the other " +  str(sum(coef == 0)) + " variables")
        selected_features = np.array(coef[coef != 0].axes[0])
        return selected_features
     '''  