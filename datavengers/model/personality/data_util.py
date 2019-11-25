import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler

class Data_Util:
    
    def __init__(self):
        pass
    
    
    # Useful for having Matrices X with targets
    def build_relations_df(self, users_col, rel_df, n):
        users_encoding = {}
        # Encoding users Ids
        for i, user in enumerate(users_col.to_numpy()):
            users_encoding[user] = i
        
        arr_k = (rel_df['like_id'].value_counts().keys()).to_numpy()
        #arr_v = np.array(rel_df['like_id'].value_counts().to_list())
        
        selected_k = arr_k
        if n > 0:
            selected_k = arr_k[:n]
        new_data = np.zeros((len(users_encoding), len(selected_k))).astype(int)
        for i, k in enumerate(selected_k):
            users_cols = rel_df.loc[rel_df['like_id'] == k]
            users_found = users_cols[users_cols.columns[1]]
            for user in users_found:
                new_data[users_encoding[user], i] += 1
        
        # Creating a data frame
        new_df = pd.DataFrame(new_data, columns = selected_k) 
        new_df.insert(0, 'userId', users_col, allow_duplicates=False)
        
        # End
        return new_df
    
    def combine_nrc_liwc_rel(self, nrc, liwc, rel, columns_to_remove = [], transform = None):
        feat_combined =  pd.merge(nrc, liwc, on='userId', how='inner')
        feat_combined =  pd.merge(feat_combined, rel, on='userId', how='inner')
        feat_combined = feat_combined.drop(['userId'], axis=1)
        if len(columns_to_remove) != 0:
            feat_combined = feat_combined.drop(columns_to_remove, axis=1)
        
        data = feat_combined.to_numpy()    
        size_to_scale = len(nrc.columns) + len(liwc.columns) - 1
        if transform == 'normalize':
            data[:, :size_to_scale] = normalize(data[:, :size_to_scale])
        elif transform == 'scale':
            scaler = MinMaxScaler(feature_range=[0, 1])
            data[:, :size_to_scale] = scaler.fit_transform(data[:, :size_to_scale])
        elif transform == 'standardize':
            scaler = StandardScaler()
            data[:, :size_to_scale] = scaler.fit_transform(data[:, :size_to_scale])
        return data 
        


    # Useful for having Matrices X with targets
    def build_df_with_target(self, df_feats, df_targets, target_name):
        buff = df_targets[['userid',str(target_name)]]
        buff.columns = ['userId',str(target_name)]
        new_df = pd.merge(df_feats, buff, on='userId', how='inner')
        return new_df.drop(['userId'], axis = 1)
        
        
    def combine_df_feats_with_transformation(self, df_feats_1, df_feats_2, columns_to_remove = [], transform = None):
        feat_combined =  pd.merge(df_feats_1, df_feats_2, on='userId', how='inner')
        feat_combined = feat_combined.drop(['userId'], axis=1)
        if len(columns_to_remove) != 0:
            feat_combined = feat_combined.drop(columns_to_remove, axis=1)
        data = feat_combined.to_numpy()       
        if transform == 'normalize':
            data = normalize(data)
        elif transform == 'scale':
            scaler = MinMaxScaler(feature_range=[0.0, 1.0])
            data = scaler.fit_transform(data)
        elif transform == 'standardize':
            scaler = StandardScaler()
            data = scaler.fit_transform(data)
        return data 
    
    def combine_selected_features_with_transformation(self, df_feats_1, df_feats_2, columns_to_keep = [], transform = None):
        feat_combined =  pd.merge(df_feats_1, df_feats_2, on='userId', how='inner')
        feat_combined = feat_combined.drop(['userId'], axis=1)
        if len(columns_to_keep) != 0:
            feat_combined = feat_combined[columns_to_keep]
        data = feat_combined.to_numpy()       
        if transform == 'normalize':
            data = normalize(data)
        elif transform == 'scale':
            scaler = MinMaxScaler(feature_range=[0.0, 1.0])
            data = scaler.fit_transform(data)
        elif transform == 'standardize':
            scaler = StandardScaler()
            data = scaler.fit_transform(data)
        return data
    
    def get_feats(self, feats, columns_to_keep = [], transform = None):
        new_feats = feats.drop(['userId'], axis=1)
        if len(columns_to_keep) != 0:
            new_feats = new_feats[columns_to_keep]
        new_feats = new_feats.to_numpy()   
    
    def extract_targets(self, df_profiles):
        targets_df = df_profiles[['ope','neu', 'ext', 'agr', 'con']]
        return targets_df.to_numpy()
    
    def get_feats(self, feats, columns_to_remove = [], transform = None):
        new_feats = feats.drop(['userId'], axis=1)
        if len(columns_to_remove) != 0:
            new_feats = new_feats.drop(columns_to_remove, axis = 1)
        new_feats = new_feats.to_numpy()       
      
        if transform == 'normalize':
            new_feats = normalize(new_feats)
        elif transform == 'scale':
            scaler = MinMaxScaler(feature_range=[0.0, 1.0])
            new_feats = scaler.fit_transform(new_feats)
        elif transform == 'standardize':
            scaler = StandardScaler()
            new_feats = scaler.fit_transform(new_feats)
        
        return new_feats
    