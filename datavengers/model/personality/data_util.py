import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler

class Data_Util:
    
    def __init__(self):
        self._vocab = {}
        self._words_indexes = {}
        self._words_list = {}
        self.word_embeddings = None
    
    def get_vocabulary(self, rel_data_frame):
        dic = {}
        for sentence in rel_data_frame['like_id']:
            for word in sentence:
                if word in dic:
                    dic[word] += 1
                else:
                    dic[word] = 1
        return dic
    
    
    def get_n_most_common(self, dic, n):
        sorted_list = sorted(dic, key=dic.get, reverse=True)
        if 0 < n < len(sorted_list):
            return sorted_list[:n]
        return sorted_list

    
    def assign_indexes(self, vocab_list):
        index_list = {}
        for k, v in enumerate(vocab_list):
            index_list[v] = k
        return index_list
    
    def get_word_context(self, tokenized_sentence, index, window):
        ret=[]
        size = len(tokenized_sentence)
        for i in range(max(index - window, 0), min(index + window + 1, size)):
            if (i == index):
              continue
            ret.append(tokenized_sentence[i])
        return ret
    
    def filter_dataframe(self, rel, most_common):
        new_rel = rel.copy()
        for i, sentence in enumerate(new_rel['like_id']):
            filtered_list = []
            for word in sentence:
                if word in most_common:
                    filtered_list.append(word)
            new_rel['like_id'][i] = filtered_list
        return new_rel

    # Function to compute the co-occurrence matrix
    def co_occ_mat(self, X, list_words, window_size = 1000):
      # instantiating co-occ matrix
      mat = np.zeros((len(list_words), len(list_words))).astype(float)
      for i, tokenized_sentence in enumerate(X):
        for j in range(len(tokenized_sentence)):
          central_word = tokenized_sentence[j]
          if central_word not in list_words:
            continue
          context = self.get_context(tokenized_sentence, j, window_size)
          # Filtering context
          context_list = []
          for word in context:
            if word in list_words:
              context_list.append(word)
          # Updating the matrix
          for word in context_list:
            mat[list_words[central_word], list_words[word]] += 1
      # End
      return mat

    
    # Builds embeddings for relations df
    '''
    def build_relations_df(self, rel_df, max_features, profile_df, n_components, for_train = True):
        # Step 1: convert to strings
        relations  = rel_df.copy()
        relations['like_id'] = rel_df['like_id'].astype(str)
        # Step 2: Build sequences of pages likes
        merged_relations = relations.groupby('userid')['like_id'].apply((lambda x: "%s" % ' '.join(x))).reset_index()
        merged_relations['like_id'] = merged_relations['like_id'].apply(lambda x: x.split())
        if for_train == True:
            print('Creating vocabulary...')
            self._vocab = self.get_vocabulary(merged_relations)
            sorted_list = sorted(self._vocab, key=self._vocab.get, reverse=True)
            print('Found %d pages...' %len(sorted_list))
            print('Picking up the most common pages...')
            self._words_list = sorted_list[:max_features]
            self._words_indexes = self.assign_indexes(self._words_list)
            print('Filtering the dataframe...')
            new_rel = self.filter_dataframe(merged_relations, self._words_list)
            print('Creating the co-occurrence matrix...')
            mat = self.co_occ_mat(new_rel['like_id'].to_numpy(), self._words_indexes , window_size = 1000)
            transformer = TruncatedSVD(n_components)
            svd_mat = transformer.fit_transform(mat)
            print('Calculating the words embeddings...')
            
        # End
        return merged_relations
    '''
    
     
    # Renames the profile's userId column
    def preprocess_profile_df(self, profile_df):
        return profile_df.rename(columns={('userid'): ('userId')}, inplace=False)
    
    # Aligns features with userId column
    def align_features_df(self, feat_df, reference):
        return pd.merge(reference, feat_df, on='userId', how='inner')
    
    # Gets targets
    def extract_targets_df(self, profile_df, reference):
        return pd.merge(reference, profile_df[['userId', 'ope', 'neu', 'ext', 'agr', 'con']], on='userId', how='inner')
    
    # Removes userId column and returns a numpy array
    def extract_data(self, df):
        return (df.drop(['userId'], axis = 1)).to_numpy()
    
    # Scales features within a dataframe
    def scale_features(self, df, scale = ''):
        std_scaler = StandardScaler()
        min_max_Scaler = MinMaxScaler()
        transform = None
        if scale == 'minmax':
            transform = min_max_Scaler.fit_transform
        elif scale == 'normalize':
            transform = normalize
        elif scale == 'standardize':
            transform = std_scaler.fit_transform
        else:
            return df
        #Creating a copy
        new_df = df.copy()
        for col in df.columns:
            if col == 'userId':
                continue
            new_df[col] = transform(new_df[[col]])
        return new_df
    
    # Combines features of a dataframe
    def combine_features(self, feats, reference):
        main_df = reference.copy()
        for df in feats:
            main_df = pd.merge(main_df, df, on='userId', how='inner')
        return main_df
