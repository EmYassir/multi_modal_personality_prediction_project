import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, normalize, StandardScaler
from sklearn.decomposition import TruncatedSVD

class Data_Util:
    
    def __init__(self):
        self._vocab = {}
        self._words_indexes = {}
        self._words_list = {}
        self.word_embeddings = None
    
    # Gets all the pages Ids
    def get_vocabulary(self, rel_data_frame):
        dic = {}
        for sentence in rel_data_frame['like_id']:
            for word in sentence:
                if word in dic:
                    dic[word] += 1
                else:
                    dic[word] = 1
        return dic
    
    # Gets the n most common(liked) pages Ids
    def get_n_most_common(self, dic, n):
        sorted_list = sorted(dic, key=dic.get, reverse=True)
        if 0 < n < len(sorted_list):
            return sorted_list[:n]
        return sorted_list

    # Assigns unique index to each liked page id in the vocabulary
    def assign_indexes(self, vocab_list):
        index_list = {}
        for k, v in enumerate(vocab_list):
            index_list[v] = k
        return index_list
    
    # Gets the context (closes liked pages Ids) given a window
    def get_word_context(self, tokenized_sentence, index, window):
        ret=[]
        size = len(tokenized_sentence)
        for i in range(max(index - window, 0), min(index + window + 1, size)):
            if (i == index):
              continue
            ret.append(tokenized_sentence[i])
        return ret
    
    # Keeps only the liked pages that are in the vocabulary
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

    # Generates the words embeddings
    def create_dense_representation(self, rel, words_index, embeddings):
        mat = np.zeros((new_rel['like_id'].shape[0], embeddings.shape[1])).astype(float)
        for i, elem in enumerate(new_rel['like_id']):
            for word in elem:
                if word not in words_index:
                    continue
                # Averaging words representations
                mat[i, :] += embeddings[words_index[word], :]
            mat[i, :] /= len(elem)
        
        # Creating dataframe columns
        new_columns = ['userId']
        for i in range(embeddings.shape[1]):
            new_columns.append('component_' + str(i))
        # Creating dataframe
        new_data = np.column_stack((new_rel['like_id'], mat))
        return pd.DataFrame(new_data, columns=new_columns)
    
    # Builds embeddings for relations df
    def build_relations_df(self, rel_df, max_features, profile_df, n_components, transformer, for_train = True):
        # Step 1: convert to strings
        relations  = rel_df.copy()
        relations['like_id'] = relations['like_id'].astype(str)
        # Step 2: Build sequences of pages likes
        merged_relations = relations.groupby('userid')['like_id'].apply((lambda x: "%s" % ' '.join(x))).reset_index()
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
            embeddings = transformer.fit_transform(mat)
            print('Calculating the words embeddings...')
            dense_rep = create_dense_representation(relation_train, self._words_indexes, embeddings)
        else:
            embeddings = transformer.transform(mat)
            print('Calculating the words embeddings...')
            dense_rep = create_dense_representation(relation_train, self._words_indexes, embeddings)
        return dense_rep
    
    
     
    # Renames the profile's userId column
    def format_userid_column(self, df):
        return df.rename(columns={('userid'): ('userId')}, inplace=False)
    
    # Aligns features with userId column
    def align_features_df(self, feat_df, reference):
        return pd.merge(reference, feat_df, on='userId', how='inner')
    
    # Gets targets
    def extract_targets_df(self, profile_df, reference):
        return pd.merge(reference, profile_df[['userId', 'ope', 'neu', 'ext', 'agr', 'con']], on='userId', how='inner')
    
    # Removes userId column and returns a numpy array
    def extract_data(self, df):
        return (df.drop(['userId'], axis = 1)).to_numpy()
    
    # Returns a dataframe with  selectedfeatures
    def extract_feature_from_profile(self, profile, columns):
        list_cols = ['userId'] + columns
        new_df = profile[list_cols]
        return new_df
    
    # Scales features within a dataframe
    def scale_features(self, df, scale = ''):
        std_scaler = StandardScaler()
        min_max_Scaler = MinMaxScaler(0,1)
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
    
    # Gets preprocessed age data
    def preprocess_age_df(self, age_df):
        #{24: "xx-24", 34: "25-34", 49: "35-49", 1000: "50-xx"}
        #{0: "xx-24", 25: "25-34", 35: "35-49", 50: "50-xx"}
        new_age_df = age_df.copy()
        new_age_df['age'] = new_age_df['age'].apply(lambda x: 0.0 if x < 25.0 else (25.0 if x < 35.0 else (35.0 if x < 50.0 else 50.0)))
        return new_age_df
        
        
    
    
    
    