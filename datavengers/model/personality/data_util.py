import pandas as pd

class Data_Util:
    
    def __init__(self):
        pass
        
    # Useful for having Matrices X with targets
    def build_df_with_target(self, df_feats, df_targets, target_name):
        buff = df_targets[['userid',str(target_name)]]
        buff.columns = ['userId',str(target_name)]
        return pd.merge(df_feats, buff, on='userId', how='inner')
        
        
        