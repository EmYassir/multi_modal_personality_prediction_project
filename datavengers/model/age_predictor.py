import pandas as pd
import numpy as np
import pickle as pkl

from datavengers.model.predictor import Predictor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class AgePredictor(Predictor):

    def __init__(self):
        super(AgePredictor, self).__init__()
        self._age_categories = {24: "xx-24", 34: "25-34", 49: "35-49", 1000: "50-xx"}
        self._img_clf = None
        self._rel_clf = None
        self._txt_clf = None
        self._n_groups = None
        self._like_idx_dict = None

    def _convert_age_to_category(self, age):
        for limit, category in self._age_categories.items():
            if age <= limit:
                return category

    def _find_lines_intersection(self, a, b, c, d):
        # Line AB represented as a1x + b1y = c1
        a1 = b[1] - a[1]
        b1 = a[0] - b[0]
        c1 = a1 * (a[0]) + b1 * (a[1])
        # Line CD represented as a2x + b2y = c2
        a2 = d[1] - c[1]
        b2 = c[0] - d[0]
        c2 = a2 * (c[0]) + b2 * (c[1])
        determinant = a1 * b2 - a2 * b1
        x = (b2 * c1 - b1 * c2) / determinant
        y = (a1 * c2 - a2 * c1) / determinant

        return x, y

    def _find_points_distance(self, a, b):
        return np.sqrt((b[0] - a[0]) ** 2 + ((b[1] - a[1]) ** 2))

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

    def _preprocess_img_data(self, raw_data, is_test=True):
        oxford_data = raw_data.get_oxford()
        a = oxford_data["eyeLeftOuter_x"], oxford_data["eyeLeftOuter_y"]
        b = oxford_data["eyeLeftInner_x"], oxford_data["eyeLeftInner_y"]
        c = oxford_data["eyeLeftTop_x"], oxford_data["eyeLeftTop_y"]
        d = oxford_data["eyeLeftBottom_x"], oxford_data["eyeLeftBottom_y"]
        left_eye_pos_x, left_eye_pos_y = self._find_lines_intersection(a, b, c, d)
        a = oxford_data["eyeRightOuter_x"], oxford_data["eyeRightOuter_y"]
        b = oxford_data["eyeRightInner_x"], oxford_data["eyeRightInner_y"]
        c = oxford_data["eyeRightTop_x"], oxford_data["eyeRightTop_y"]
        d = oxford_data["eyeRightBottom_x"], oxford_data["eyeRightBottom_y"]
        right_eye_pos_x, right_eye_pos_y = self._find_lines_intersection(a, b, c, d)
        eye_to_eye_distance = self._find_points_distance((left_eye_pos_x, left_eye_pos_y), (right_eye_pos_x, right_eye_pos_y))
        eye_to_nose_distance = self._find_points_distance((right_eye_pos_x, right_eye_pos_y), (oxford_data["noseTip_x"], oxford_data["noseTip_y"]))
        eye_to_lip_distance = self._find_points_distance((right_eye_pos_x, right_eye_pos_y), (oxford_data["underLipBottom_x"], oxford_data["underLipBottom_y"]))
        raw_data._oxford = oxford_data.assign(
            eyeLeft_x=left_eye_pos_x,
            eyeLeft_y=left_eye_pos_y,
            eyeRight_x=right_eye_pos_x,
            eyeRight_y=right_eye_pos_y,
            eyeToEyeDistance=eye_to_eye_distance,
            eyeToNoseDistance=eye_to_nose_distance,
            eyeToLipDistance=eye_to_lip_distance,
            ete_etn_ratio=eye_to_eye_distance/eye_to_nose_distance,
            ete_etl_ratio=eye_to_eye_distance/eye_to_lip_distance,
            etn_etl_ratio=eye_to_nose_distance/eye_to_lip_distance
        )

        X_img = raw_data.get_oxford().drop_duplicates(subset="userId", keep="first", inplace=False)

        print("Preprocessing image data, done")

        if is_test:
            return X_img.loc[:, "eyeToEyeDistance":]
        else:
            return X_img.loc[:, "eyeToEyeDistance":], self._get_label_values(raw_data, X_img["userId"])

    def _preprocess_relational_data(self, raw_data, is_test=True, group_size=100):
        relational_ds = raw_data.get_relation()
        user_ids = relational_ds["userid"].unique()
        like_ids = relational_ds["like_id"].unique()
        if not is_test:
            self._n_groups = like_ids.shape[0] // group_size
            self._like_idx_dict = {l: (i // group_size) for i, l in enumerate(like_ids)}
        users_dict = {u: np.zeros(self._n_groups) for u in user_ids}

        for user_id, like_id in zip(relational_ds["userid"], relational_ds["like_id"]):
            ul = users_dict[user_id]
            try:
                ul[self._like_idx_dict[like_id] - 1] += 1
            except KeyError:
                continue

        X_rel = pd.DataFrame.from_dict(users_dict, orient='index')

        print("Preprocessing relational data, done")

        if is_test:
            return X_rel
        else:
            return X_rel, self._get_label_values(raw_data, X_rel.index.values)

    def _preprocess_text_data(self, raw_data, is_test=True):
        liwc_data = raw_data.get_liwc()
        user_ids = liwc_data["userId"].unique()

        print("Preprocessing text data, done")

        if is_test:
            return liwc_data.loc[:, "Sixltr":]
        else:
            return liwc_data.loc[:, "Sixltr":], self._get_label_values(raw_data, user_ids)

    def train(self, raw_train_data):
        raw_train_data = self._preprocess_age_group_labels(raw_train_data)
        # X_img, y_img = self._preprocess_img_data(raw_train_data, is_test=False)
        X_rel, y_rel = self._preprocess_relational_data(raw_train_data, is_test=False)
        X_txt, y_txt = self._preprocess_text_data(raw_train_data, is_test=False)
        # self._img_clf = KNeighborsClassifier(n_neighbors=100, weights="uniform")
        self._rel_clf = RandomForestClassifier(n_estimators=20, bootstrap=False, criterion="entropy", max_depth=10,
                                         max_features=2000, min_samples_split=10)
        self._txt_clf = LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=1, C=0.01, max_iter=5000)
        # self._img_clf.fit(X_img, y_img)
        self._rel_clf.fit(X_rel, y_rel)
        self._txt_clf.fit(X_txt, y_txt)

        print("Training done")

    def predict(self, raw_test_data, majority_class="xx-24"):
        # X_img = self._preprocess_img_data(raw_test_data)
        X_rel = self._preprocess_relational_data(raw_test_data)
        X_txt = self._preprocess_text_data(raw_test_data)
        # img_preds = pd.DataFrame(self._img_clf.predict(X_img))
        rel_preds = self._rel_clf.predict(X_rel)
        txt_preds = self._txt_clf.predict(X_txt)

        # for i, (r, t) in enumerate(zip(rel_preds, txt_preds)):
        #     if t == majority_class:
        #         if r != majority_class:
        #             txt_preds[i] = r
        # txt_preds.to_csv(path_or_buf="/home/alexpehpeh/PycharmProjects/IFT6758_Project/datavengers/persistence/res.csv")
        return txt_preds

    def load_model(self, location="./datavengers/persistence/age/age_predictor.model"):
        with open(location, 'rb') as fd:
            clazz = pkl.load(fd)
            self._img_clf = clazz._img_clf
            self._rel_clf = clazz._rel_clf
            self._txt_clf = clazz._txt_clf
            self._n_groups = clazz._n_groups
            self._like_idx_dict = clazz._like_idx_dict

    def save_model(self, location="./datavengers/persistence/age/age_predictor.model"):
        with open(location, 'wb') as fd:
            pkl.dump(self, fd)


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
