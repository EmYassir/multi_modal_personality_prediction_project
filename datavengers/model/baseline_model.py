import pandas as pd
import xml.etree.ElementTree as et
import pickle


class BaselineModel:

    def __init__(self, train_data_set, test_data_set, output_folder):
        self.data_set = train_data_set
        self.test_set = test_data_set
        self.output_folder = output_folder
        self.age_categories = {24: "xx-24", 34: "25-34", 49: "35-49", 1000: "50-xx"}
        self.gender_categories = ["male", "female"]
        self.data_set, self.age_pred, self.gender_pred, self.ope_pred = None, None, None, None
        self.con_pred, self.ext_pred, self.agr_pred, self.neu_pred = None, None, None, None
        self.profile_file_name = "/Profile/Profile.csv"

    def convert_age_to_category(self, age):
        for limit, category in self.age_categories.items():
            if age <= limit:
                return category

    def convert_gender_to_category(self, gender):
        if gender == 0:
            return self.gender_categories[0]
        else:
            return self.gender_categories[1]

    def write_to_xml(self, user_id):
        user = et.Element("user", {
            "id": user_id,
            "age_group": self.age_pred,
            "gender": self.gender_pred,
            "extrovert": str(self.ext_pred),
            "neurotic": str(self.neu_pred),
            "agreeable": str(self.agr_pred),
            "conscientious": str(self.con_pred),
            "open": str(self.ope_pred)
        })
        file = open(f"{self.output_folder}/{user_id}.xml", "w")
        file.write(et.tostring(user, encoding="unicode"))

    def dump(self, filename):
        out = open(filename, "wb")
        pickle.dump(self, out)
        out.close()

    def load(self, filename=None):
        inf = open(filename, "rb")
        model = pickle.load(inf)
        inf.close()
        self.age_pred = model.age_pred
        self.gender_pred = model.gender_pred
        self.ope_pred = model.ope_pred
        self.con_pred = model.con_pred
        self.ext_pred = model.ext_pred
        self.agr_pred = model.agr_pred
        self.neu_pred = model.neu_pred

    def train(self):
        # self.data_set = pd.read_csv(f"{self.train_data_path}/{self.profile_file_name}")
        self.data_set["gender"] = self.data_set["gender"].apply(self.convert_gender_to_category)
        self.data_set = self.data_set.assign(
            age_group=lambda df: self.data_set["age"].apply(self.convert_age_to_category)
        )
        # Baseline predictions
        count_age = self.data_set.groupby(["age_group"]).size()
        self.age_pred = count_age.idxmax()
        count_gender = self.data_set.groupby(["gender"]).size()
        self.gender_pred = count_gender.idxmax()
        self.ope_pred = self.data_set["ope"].mean()
        self.con_pred = self.data_set["con"].mean()
        self.ext_pred = self.data_set["ext"].mean()
        self.agr_pred = self.data_set["agr"].mean()
        self.neu_pred = self.data_set["neu"].mean()

    def compute_predictions(self):
        # test_data_set = pd.read_csv(f"{self.test_data_path}/{self.profile_file_name}")
        self.test_set["userid"].apply(self.write_to_xml)
