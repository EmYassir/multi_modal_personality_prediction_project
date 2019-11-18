import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler

class Data_Util:
    
    def __init__(self):
        pass
        
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
    