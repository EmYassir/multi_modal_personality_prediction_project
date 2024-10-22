import numpy as np
import pickle as pkl
import logging
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from keras.optimizers import Adam
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from datavengers.model.data import Data
from datavengers.model.predictor import Predictor
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight
from datavengers.model.relation_dictionary import RelationDictionary
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.models import model_from_json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgePredictor(Predictor):
    """
    This class includes all of the training and prediction logic for the age prediction task

    Attributes
    ----------
    _age_categories : list
        a list containing the age group labels to be used for prediction
    _page_dict : dict
        a dictionary containing the "vocabulary" generated from the page ids in the relational data
    _selected_pages : dict
        a dictionary containing all of the page ids that were selected according to the criteria defined in the logic
    _vectorizer : sklearn.feature_extraction.text.TfidfVectorizer
        a TF-IDF vectorizer class which is fit on the training data. Used afterwards to vectorize test data for predictions
    _label_encoder : sklearn.preprocessing.LabelEncoder
        a label encoder which transforms the labels in _age_categories to int-coded labels
    _binarizer : sklearn.preprocessing.MultiLabelBinarizer
        a label binarizer which generates a one-hot representation of a label
    _k_best_model : sklearn.feature_selection.SelectKBest
        a select k best model fit on the training data of the LIWC set

    Methods
    -------
    _convert_age_to_category(age)
        Converts an age in int format to its corresponding label in _age_categories
    _get_label_values(raw_data, user_ids)
        Produces a list of age group labels for the user ids in user_ids
    _preprocess_age_group_labels(raw_data)
        Adds the age_group attribute to the raw_data.profiles data set
    _update_page_dict(raw_data, grouped_data)
        Creates and updates the page ids dictionary from the relational data
    _preprocess_relational_data(raw_data, is_test=False)
        Pre-processes the relational data to generate the text features per user formed by one sentence with its page likes
    _preprocess_text_data(raw_data, is_test=False)
        Pre-processes the LIWC data to select the 60 best features by the chi2 criterion
    _preprocess_relational_text_data(raw_data, is_test=False)
        Pre-processes both LIWC and relational data and generates a concatenated matrix of both data sets
    train(raw_train_data)
        Creates a model a trains it by using raw_train_data
    predict(raw_test_data)
        Uses the created model to make predictions from raw_test_data
    load(location)
        Loads the model and class attributes from location
    save(classifier, location)
        Saves the trained classifier and class attributes in location
    """

    default_location = './datavengers/persistence/age/'  # official
    # default_location = '../persistence/age/'  # local

    def __init__(self):
        super(AgePredictor, self).__init__()
        self._age_categories = {24: "xx-24", 34: "25-34", 49: "35-49", 1000: "50-xx"}
        self._page_dict = None
        self._selected_pages = None
        self._vectorizer = None
        self._label_encoder = None
        self._binarizer = None
        self._k_best_model = None

    def _convert_age_to_category(self, age):
        """
        Converts an age in int format to its corresponding label in _age_categories

        Parameters
        ----------
        age : int
            The age of a user in the data

        Returns
        ------
        str : label of age group corresponding to age in _age_categories
        """
        for limit, category in self._age_categories.items():
            if age <= limit:
                return category

    def _get_label_values(self, raw_data, user_ids):
        """
        Produces a list of age group labels for the user ids in user_ids

        Parameters
        ----------
        raw_data : datavengers.model.data.Data
            Object containing the oxford, profile, relational and liwc data sets
        user_ids: list
            List of user ids

        Returns
        ------
        list : age group labels corresponding to the user list in user_ids
        """
        y = []
        profile_data = raw_data.get_profiles()
        for user_id in user_ids:
            y.append(profile_data.loc[profile_data.userid == user_id, "age_group"].iloc[0])

        return y

    def _preprocess_age_group_labels(self, raw_data):
        """
        Adds the age_group attribute to the raw_data.profiles data set

        Parameters
        ----------
        raw_data : datavengers.model.data.Data
            Object containing the oxford, profile, relational and liwc data sets

        Returns
        ------
        datavengers.model.data.Data : data object to which the age_group attribute was added
        """
        raw_data._profiles = raw_data.get_profiles().assign(
            age_group=lambda df: df["age"].apply(self._convert_age_to_category)
        )
        return raw_data

    def _update_page_dict(self, raw_data, grouped_data):
        """
        Creates and updates the page ids dictionary from the relational data

        Parameters
        ----------
        raw_data : datavengers.model.data.Data
            Object containing the oxford, profile, relational and liwc data sets
        grouped_data: pandas.DataFrame
            DataFrame containing the sorted relational data by user id
        """
        labels = self._get_label_values(raw_data, grouped_data['userid'].values)
        sorted_labels = np.unique(labels)
        self._page_dict = RelationDictionary(sorted_labels)
        logger.info('### Updating page dictionary')
        for page_string, label in zip(grouped_data['like_id'].values, labels):
            self._page_dict.update(page_string, label)
        # Select pages
        self._selected_pages = {}
        for l in sorted_labels:
            u_list = self._page_dict.get_n_pages_unique_to_label(l, 20000)
            # u_list = self._page_dict.get_n_top_pages_per_label(l, 20000)
            for u in u_list:
                self._selected_pages[u] = True

    def _preprocess_relational_data(self, raw_data, is_test=False):
        """
        Pre-processes the relational data to generate the text features per user formed by one sentence with its page likes

        Parameters
        ----------
        raw_data : datavengers.model.data.Data
            Object containing the oxford, profile, relational and liwc data sets
        is_test: bool
            Flag which indicates if the dictionary has to be updated from the raw_data.
            Should be only set to True when in test mode

        Returns
        ------
        (list, list, list) : tuple containing the processed data (text features per user), the labels, and the user ids
        """
        relational_ds = raw_data.get_relation()
        relational_ds['like_id'] = relational_ds['like_id'].astype(str)
        grouped_data = relational_ds.groupby('userid')['like_id'].apply((lambda x: "%s" % ' '.join(x))).reset_index()
        grouped_data = grouped_data.sort_values(by='userid')
        if not is_test:
            self._update_page_dict(raw_data, grouped_data)
        logger.info("### Filtering pages")
        processed_data = []
        for page_string in grouped_data['like_id'].values:
            tokens = page_string.split(' ')
            processed_data.append(' '.join([p for p in tokens if p in self._selected_pages]))
        labels = None if is_test else self._get_label_values(raw_data, grouped_data['userid'].values)

        return processed_data, labels, grouped_data['userid'].values

    def _preprocess_text_data(self, raw_data, is_test=False):
        """
        Pre-processes the LIWC data to select the 60 best features by the chi2 criterion

        Parameters
        ----------
        raw_data : datavengers.model.data.Data
            Object containing the oxford, profile, relational and liwc data sets
        is_test: bool
            Flag which indicates if the select k best model has to be fit to raw_data
            Should be only set to True when in test mode

        Returns
        ------
        (list, list, list) : tuple containing the processed data, the labels, and the user ids
        """
        liwc_data = raw_data.get_liwc()
        liwc_data = liwc_data.sort_values(by='userId')
        user_ids = liwc_data["userId"].unique()
        logger.info("### Pre-processing text data")
        labels = None if is_test else self._get_label_values(raw_data, user_ids)
        if not is_test:
            self._k_best_model = SelectKBest(chi2, k=60).fit(liwc_data.loc[:, "Sixltr":].values, labels)
        x_text = self._k_best_model.transform(liwc_data.loc[:, "Sixltr":].values)

        return x_text, labels, user_ids

    def _preprocess_relational_text_data(self, raw_data, is_test=False):
        """
        Pre-processes both LIWC and relational data and generates a concatenated matrix of both data sets

        Parameters
        ----------
        raw_data : datavengers.model.data.Data
            Object containing the oxford, profile, relational and liwc data sets
        is_test: bool
            Flag which indicates if the select k best model has to be fit to raw_data
            Should be only set to True when in test mode

        Returns
        ------
        (list, list, list) : tuple containing the processed data, the labels (if is_test, labels=None), and the user ids
        """
        x_text, y_text, user_ids_t = self._preprocess_text_data(raw_data, is_test)
        x_rel, y_rel, user_ids_r = self._preprocess_relational_data(raw_data, is_test)
        assert y_text == y_rel, 'LIWC labels different to relational labels'
        assert user_ids_t.all() == user_ids_r.all(), 'LIWC userIds different to relational userIds'
        logger.info("### Vectorizing")
        if not is_test:
            self._vectorizer = TfidfVectorizer()
            self._vectorizer.fit(x_rel)
        x_mix = self._vectorizer.transform(x_rel)
        logger.info("### Finished vectorizing")
        x_mix = np.concatenate((x_mix.todense(), x_text), axis=1)
        logger.info(f"### x_mix shape {x_mix.shape}")

        return (x_mix, None, user_ids_r) if is_test else (x_mix, y_rel, user_ids_r)

    def train(self, raw_train_data):
        """
        Creates a model a trains it by using raw_train_data

        Parameters
        ----------
        raw_train_data : datavengers.model.data.Data
            Object containing the oxford, profile, relational and liwc data sets

        """
        logger.info("### Starting training")
        raw_train_data = self._preprocess_age_group_labels(raw_train_data)
        x_train, y_train, _ = self._preprocess_relational_text_data(raw_train_data)
        classifier = Sequential()
        # First Hidden Layer
        classifier.add(Dense(32, activation='relu', kernel_initializer='random_normal', input_dim=x_train.shape[1]))
        # Dropout
        classifier.add(Dropout(rate=0.1))
        # Output Layer
        classifier.add(Dense(4, activation='softmax', kernel_initializer='random_normal'))
        self._label_encoder = LabelEncoder()
        fit_y_train = self._label_encoder.fit_transform(y_train)
        fit_y_train = [item for item in fit_y_train.astype(str)]
        self._binarizer = MultiLabelBinarizer()
        oh_y_train = self._binarizer.fit_transform(fit_y_train)
        # Compiling the neural network
        opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
        classifier.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        # Fitting the data to the training dataset
        classifier.fit(x_train, oh_y_train, batch_size=64, epochs=30, sample_weight=compute_sample_weight('balanced', fit_y_train))
        logger.info("### Training done")
        self.save(classifier)

    def predict(self, raw_test_data):
        """
        Uses the created model to make predictions from raw_test_data

        Parameters
        ----------
        raw_test_data : datavengers.model.data.Data
            Object containing the oxford, profile, relational and liwc data sets

        Returns
        ------
        list : list of predictions obtained from raw_test_data
        """
        logger.info("### Starting predictions")
        classifier = self.load()
        x_test, _, user_ids = self._preprocess_relational_text_data(raw_test_data, is_test=True)
        preds = classifier.predict(x_test)
        preds = np.argmax(preds, axis=1)
        logger.info("### Predictions done")
        preds = self._label_encoder.inverse_transform(preds)
        y_dic = {}
        for u, p in zip(user_ids, preds):
            y_dic[u] = p

        return y_dic

    def load(self, location=default_location):
        """
        Loads the model and class attributes from location

        Parameters
        ----------
        location : str, optional
            Location from which the model is loaded

        Returns
        ------
        object : the classifier loaded from location
        """
        logger.info("### Loading model")
        class_file = f'{location}age_predictor.model'
        classifier_file = f'{location}model.json'
        weights_file = f'{location}model.h5'
        with open(class_file, 'rb') as fd:
            clazz = pkl.load(fd)
            self._page_dict = clazz._page_dict
            self._selected_pages = clazz._selected_pages
            self._vectorizer = clazz._vectorizer
            self._label_encoder = clazz._label_encoder
            self._binarizer = clazz._binarizer
            self._k_best_model = clazz._k_best_model
        json_file = open(classifier_file, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        classifier = model_from_json(loaded_model_json)
        classifier.load_weights(weights_file)
        opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
        classifier.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        logger.info("### Model loaded")

        return classifier

    def save(self, classifier, location=default_location):
        """
        Saves the trained classifier and class attributes in location

        Parameters
        ----------
        classifier : object
            Classifier to save
        location : str, optional
            Location to which the model is saved
        """
        logger.info("### Saving model")
        class_file = f'{location}age_predictor.model'
        classifier_file = f'{location}model.json'
        weights_file = f'{location}model.h5'
        with open(class_file, 'wb') as fd:
            pkl.dump(self, fd)
        model_json = classifier.to_json()
        with open(classifier_file, "w") as json_file:
            json_file.write(model_json)
        classifier.save_weights(weights_file)
        logger.info("### Model saved")


# Utility function to report best scores
# def report(results, n_top=3):
#     for i in range(1, n_top + 1):
#         candidates = np.flatnonzero(results['rank_test_score'] == i)
#         for candidate in candidates:
#             print("Model with rank: {0}".format(i))
#             print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
#                   results['mean_test_score'][candidate],
#                   results['std_test_score'][candidate]))
#             print("Parameters: {0}".format(results['params'][candidate]))
#             print("")
#
#
# def test_model(model_name, model, X_train, X_test, y_train, y_test):
#     logger.info('########## Testing ' + str(model_name) + ' ##########')
#     logger.info('### Training ...')
#     model.fit(X_train, y_train)
#     logger.info('### Predicting ...')
#     y_pred_train = model.predict(X_train)
#     y_pred = model.predict(X_test)
#     logger.info('-> training accuracy %s' % accuracy_score(y_pred_train, y_train))
#     logger.info('-> test accuracy %s' % accuracy_score(y_pred, y_test))
#     return y_pred


# if __name__ == "__main__":
#     train_data = Data("/home/alexpehpeh/PycharmProjects/IFT6758_Project/data/Train/liwc.csv",
#                       "/home/alexpehpeh/PycharmProjects/IFT6758_Project/data/Train/nrc.csv",
#                       "/home/alexpehpeh/PycharmProjects/IFT6758_Project/data/Train/Relation.csv",
#                       "/home/alexpehpeh/PycharmProjects/IFT6758_Project/data/Train/oxford.csv",
#                       "/home/alexpehpeh/PycharmProjects/IFT6758_Project/data/Train/Profile.csv")
#     test_data = Data("/home/alexpehpeh/PycharmProjects/IFT6758_Project/data/Public_Test/Text/liwc.csv",
#                       "/home/alexpehpeh/PycharmProjects/IFT6758_Project/data/Public_Test/Text/nrc.csv",
#                       "/home/alexpehpeh/PycharmProjects/IFT6758_Project/data/Public_Test/Relation/Relation.csv",
#                       "/home/alexpehpeh/PycharmProjects/IFT6758_Project/data/Public_Test/Image/oxford.csv",
#                       "/home/alexpehpeh/PycharmProjects/IFT6758_Project/data/Public_Test/Profile/Profile.csv")
    # ap = AgePredictor()
    # age_predictor.train(train_data)
    # preds = age_predictor.predict(train_data)
    # print(preds)
    # raw_train_data = ap._preprocess_age_group_labels(train_data)
    # x_train, y_train, _ = ap._preprocess_relational_text_data(train_data)
    # x_train, y_train, x_val, y_val = train_test_split(x_train, y_train, test_size=.10, stratify=y_train, shuffle=True)
    # model = LogisticRegression(
    #     penalty='l2', class_weight='balanced', solver='sag', multi_class='multinomial', verbose=True
    # )
    # test_model('LR', model, x_train, x_val, y_train, y_val)



