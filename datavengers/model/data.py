import pandas as pd

class Data:
    
    def __init__(self, csv_liwc, csv_nrc, csv_relation, csv_oxford, csv_profiles):
        self._liwc = pd.read_csv(csv_liwc)
        self._nrc = pd.read_csv(csv_nrc)
        self._relation = pd.read_csv(csv_relation)
        self._oxford = pd.read_csv(csv_oxford)
        self._profiles = pd.read_csv(csv_profiles)
    
    # Setters
    def load_liwc(self,csv_file) :
        self._liwc = pd.read_csv(csv_file)

    def load_nrc(self,csv_file) :
        self._nrc = pd.read_csv(csv_file)

    def load_relation(self,csv_file) :
        self._relation = pd.read_csv(csv_file)

    def load_oxford(self,csv_file) :
        self._oxford = pd.read_csv(csv_file)

    def load_profiles(self,csv_file) :
        self._profiles = pd.read_csv(csv_file)

    # Getters
    def get_liwc(self) :
        return self._liwc

    def get_nrc(self) :
        return self._nrc
    
    def get_relation(self) :
        return self._relation

    def get_oxford(self) :
        return self._oxford

    def get_profiles(self) :
        return self._profiles
