import numpy as np
import pickle as pkl
import logging

from sklearn.feature_extraction.text import TfidfVectorizer
from datavengers.model.data import Data
from datavengers.model.predictor import Predictor
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score
from datavengers.model.relation_dictionary import RelationDictionary
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.models import model_from_json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgePredictor(Predictor):

    default_location = './datavengers/persistence/age/'  # official
    # default_location = '../persistence/age/'  # local

    def __init__(self):
        super(AgePredictor, self).__init__()
        self._age_categories = {24: "xx-24", 34: "25-34", 49: "35-49", 1000: "50-xx"}
        self._img_clf = None
        self._rel_clf = None
        self._txt_clf = None
        self._n_groups = None
        self._like_idx_dict = None
        self._page_dict = None
        self._selected_pages = None
        self._vectorizer = None
        self._label_encoder = None
        self._binarizer = None

    def _convert_age_to_category(self, age):
        for limit, category in self._age_categories.items():
            if age <= limit:
                return category

    def _get_label_values(self, raw_data, user_ids):
        y = []
        profile_data = raw_data.get_profiles()
        for user_id in user_ids:
            y.append(profile_data.loc[profile_data.userid == user_id, "age_group"].iloc[0])

        return y

    def _preprocess_age_group_labels(self, raw_data):
        raw_data._profiles = raw_data.get_profiles().assign(
            age_group=lambda df: df["age"].apply(self._convert_age_to_category)
        )
        return raw_data

    def _update_page_dict(self, raw_data, grouped_data):
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

        return processed_data, labels

    def _preprocess_text_data(self, raw_data, is_test=False):
        liwc_data = raw_data.get_liwc()
        liwc_data = liwc_data.sort_values(by='userId')
        user_ids = liwc_data["userId"].unique()
        logger.info("### Pre-processing text data")
        labels = None if is_test else self._get_label_values(raw_data, user_ids)

        return liwc_data.loc[:, "Sixltr":].values, labels

    def _preprocess_relational_text_data(self, raw_data, is_test=False):
        x_text, y_text = self._preprocess_text_data(raw_data, is_test)
        x_rel, y_rel = self._preprocess_relational_data(raw_data, is_test)
        logger.info("### Vectorizing")
        if not is_test:
            self._vectorizer = TfidfVectorizer()
            self._vectorizer.fit(x_rel)
        x_mix = self._vectorizer.transform(x_rel)
        logger.info("### Finished vectorizing")
        x_mix = np.concatenate((x_mix.todense(), x_text), axis=1)
        logger.info(f"### x_mix shape {x_mix.shape}")

        return (x_mix, None) if is_test else (x_mix, y_rel)

    def train(self, raw_train_data):
        logger.info("### Starting training")
        raw_train_data = self._preprocess_age_group_labels(raw_train_data)
        x_train, y_train = self._preprocess_relational_text_data(raw_train_data)
        classifier = Sequential()
        # First Hidden Layer
        classifier.add(Dense(32, activation='relu', kernel_initializer='random_normal', input_dim=x_train.shape[1]))
        # Dropout
        classifier.add(Dropout(rate=0.1))
        # Output Layer
        classifier.add(Dense(4, activation='sigmoid', kernel_initializer='random_normal'))
        self._label_encoder = LabelEncoder()
        fit_y_train = self._label_encoder.fit_transform(y_train)
        fit_y_train = [item for item in fit_y_train.astype(str)]
        self._binarizer = MultiLabelBinarizer()
        oh_y_train = self._binarizer.fit_transform(fit_y_train)
        # Compiling the neural network
        classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        # Fitting the data to the training dataset
        classifier.fit(x_train, oh_y_train, batch_size=10, epochs=8)
        logger.info("### Training done")
        self.save(classifier)

    def predict(self, raw_test_data):
        logger.info("### Starting predictions")
        classifier = self.load()
        x_test, _ = self._preprocess_relational_text_data(raw_test_data, is_test=True)
        preds = classifier.predict(x_test)
        preds = np.argmax(preds, axis=1)
        logger.info("### Predictions done")
        return self._label_encoder.inverse_transform(preds)

    def load(self, location=default_location):
        logger.info("### Loading model")
        class_file = f'{location}age_predictor.model'
        classifier_file = f'{location}model.json'
        weights_file = f'{location}model.h5'
        with open(class_file, 'rb') as fd:
            clazz = pkl.load(fd)
            self._n_groups = clazz._n_groups
            self._like_idx_dict = clazz._like_idx_dict
            self._page_dict = clazz._page_dict
            self._selected_pages = clazz._selected_pages
            self._vectorizer = clazz._vectorizer
            self._label_encoder = clazz._label_encoder
            self._binarizer = clazz._binarizer
        json_file = open(classifier_file, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        classifier = model_from_json(loaded_model_json)
        classifier.load_weights(weights_file)
        classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        logger.info("### Model loaded")

        return classifier

    def save(self, classifier, location=default_location):
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
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


def test_model(model_name, model, X_train, X_test, y_train, y_test):
    logger.info('########## Testing ' + str(model_name) + ' ##########')
    logger.info('### Training ...')
    model.fit(X_train, y_train)
    logger.info('### Predicting ...')
    y_pred_train = model.predict(X_train)
    y_pred = model.predict(X_test)
    logger.info('-> training accuracy %s' % accuracy_score(y_pred_train, y_train))
    logger.info('-> test accuracy %s' % accuracy_score(y_pred, y_test))
    return y_pred


if __name__ == "__main__":
    train_data = Data("/home/alexpehpeh/PycharmProjects/IFT6758_Project/data/Train/liwc.csv",
                      "/home/alexpehpeh/PycharmProjects/IFT6758_Project/data/Train/nrc.csv",
                      "/home/alexpehpeh/PycharmProjects/IFT6758_Project/data/Train/Relation.csv",
                      "/home/alexpehpeh/PycharmProjects/IFT6758_Project/data/Train/oxford.csv",
                      "/home/alexpehpeh/PycharmProjects/IFT6758_Project/data/Train/Profile.csv")
    test_data = Data("/home/alexpehpeh/PycharmProjects/IFT6758_Project/data/Public_Test/Text/liwc.csv",
                      "/home/alexpehpeh/PycharmProjects/IFT6758_Project/data/Public_Test/Text/nrc.csv",
                      "/home/alexpehpeh/PycharmProjects/IFT6758_Project/data/Public_Test/Relation/Relation.csv",
                      "/home/alexpehpeh/PycharmProjects/IFT6758_Project/data/Public_Test/Image/oxford.csv",
                      "/home/alexpehpeh/PycharmProjects/IFT6758_Project/data/Public_Test/Profile/Profile.csv")
    age_predictor = AgePredictor()
    # age_predictor.train(train_data)
    preds = age_predictor.predict(train_data)
    print(preds)
    # train_data = age_predictor._preprocess_age_group_labels(train_data)
    # text_data, y_text = age_predictor._preprocess_text_data(train_data)
    # X_train, y_rel = age_predictor._preprocess_relational_data(train_data)
    # X_train, X_test, y_train, y_test = train_test_split(X_train, y_rel, stratify=y_rel, test_size=0.2, random_state=42)
    # X_text_train, X_text_test, _, _ = train_test_split(text_data, y_text, stratify=y_text, test_size=0.2, random_state=42)
    # logger.info("### Vectorizing")
    # vectorizer = TfidfVectorizer().fit(X_train)
    # X_train = vectorizer.transform(X_train)
    # X_test = vectorizer.transform(X_test)
    # X_train = np.concatenate((X_train.todense(), X_text_train), axis=1)
    # X_test = np.concatenate((X_test.todense(), X_text_test), axis=1)
    # logger.info(f"### X_train shape {X_train.shape}")
    # logger.info(f"### X_test shape {X_test.shape}")
    # logger.info("### Finished vectorizing")
    # classifier = Sequential()
    # # First Hidden Layer
    # classifier.add(Dense(32, activation='relu', kernel_initializer='random_normal', input_shape=(65574,)))
    # classifier.add(Dropout(rate=0.1))
    # # Second  Hidden Layer
    # # classifier.add(Dense(32, activation='relu', kernel_initializer='random_normal'))
    # # classifier.add(Dropout(rate=0.1))
    # # Output Layer
    # classifier.add(Dense(4, activation='sigmoid', kernel_initializer='random_normal'))
    # label_encoder = LabelEncoder()
    # int_y_train = label_encoder.fit_transform(y_train)
    # int_y_train = [item for item in int_y_train.astype(str)]
    # int_y_test = label_encoder.transform(y_test)
    # int_y_test = [item for item in int_y_test.astype(str)]
    # ml_binarizer = MultiLabelBinarizer()
    # oh_y_train = ml_binarizer.fit_transform(int_y_train)
    # oh_y_test = ml_binarizer.transform(int_y_test)
    # # Compiling the neural network
    # classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # # Fitting the data to the training dataset
    # classifier.fit(X_train, oh_y_train, batch_size=10, epochs=8)
    # y_pred = classifier.predict(X_test)
    # fp = np.argmax(y_pred, axis=1)
    # fy = [int(item) for item in int_y_test]
    # print(classification_report(label_encoder.inverse_transform(fy), label_encoder.inverse_transform(fp)))
