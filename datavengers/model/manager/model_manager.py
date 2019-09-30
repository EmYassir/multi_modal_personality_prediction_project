import pandas as pd
import xml.etree.ElementTree as et


class BaselineModel:

    def __init__(self, train_data_path, test_data_path, output_folder):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
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

    def train(self):
        self.data_set = pd.read_csv(f"{self.train_data_path}/{self.profile_file_name}")
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
        test_data_set = pd.read_csv(f"{self.test_data_path}/{self.profile_file_name}")
        test_data_set["userid"].apply(self.write_to_xml)


class ModelManager:

    def __init__(self, train_data_path, test_data_path, output_path):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.output_path = output_path
        self.baseline_model = None

    def train_all(self):
        self.baseline_model = BaselineModel(self.train_data_path, self.test_data_path, self.output_path)
        self.baseline_model.train()

    def predict_all(self):
        self.baseline_model.compute_predictions()


