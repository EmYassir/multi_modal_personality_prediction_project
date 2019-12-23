import argparse
import xml.etree.ElementTree as et

from datavengers.model.age_predictor import AgePredictor
from datavengers.model.baseline_model import BaselineModel
from datavengers.model.data import Data
from datavengers.model.gender import Gender
from datavengers.model.personality.personality import Personality


class Master:
    """
    This class orchestrates all three models to generate predictions and the respective output files

    Attributes
    ----------
    train_data : datavengers.model.data.Data
            Object containing the oxford, profile, relational and liwc data sets
    load_path : str
        path from which to load all the pre-trained models
    output_path : str
        path to which the prediction files are saved
    _baseline_model : object
        the baseline model which generates majority class predictions for classification and average for regression
    gender_prediction_model : object
        the gender prediction model
    personality_prediction_model : object
        the personality traits prediction model
    age_prediction_model : object
        the age prediction model

    Methods
    -------
    _write_to_xml(user_id, age_group, gender, ext, neu, agr, con, ope)
        Writes an xml prediction file corresponding to a particular user
    _set_baseline_model(test_data_set, pre_trained=True)
        Sets the internal baseline model
    _set_gender_model(pre_trained=True)
        Sets the internal gender prediction model
    _set_personality_model(pre_trained=True)
        Sets the internal personality prediction model
    _set_age_model(pre_trained=True)
        Sets the internal age prediction model
    get_predictions(test_data)
        Obtains gender, personality and age predictions for test_data and generates the prediction files per user
    """

    def __init__(self, train_data_path, load_path, output_path):
        """
        Parameters
        ----------
        train_data_path : str
            Path to the raw train data
        load_path : str
            Path from which the pre-trained models are loaded
        output_path : str
            Location to which the prediction files are saved
        """
        self.train_data = None
        if train_data_path is not None:
            self.train_data = Data(f"{train_data_path}/Text/liwc.csv",
                              f"{train_data_path}/Text/nrc.csv",
                              f"{train_data_path}/Relation/Relation.csv",
                              f"{train_data_path}/Image/oxford.csv",
                              f"{train_data_path}/Profile/Profile.csv")
        self.load_path = load_path
        self.output_path = output_path
        self._baseline_model = None
        self.gender_prediction_model = None
        self.personality_prediction_model = None
        self.age_prediction_model = None

    def _write_to_xml(self, user_id, age_group, gender, ext, neu, agr, con, ope):
        """
        Writes an xml prediction file corresponding to a particular user

        Parameters
        ---------
        user_id : str
            The id of the user
        age_group : str
            The age group of the user
        gender : str
            The gender of the user
        ext : float
            The extroversion score of the user
        neu : float
            The neuroticism score of the user
        agr : float
            The agreeableness score of the user
        con : float
            The conscientiousness score of the user
        ope : float
            The openness score of the user
        """
        user = et.Element("user", {
            "id": user_id,
            "age_group": age_group,
            "gender": ("male" if gender == 0 else "female"),
            "extrovert": str(ext),
            "neurotic": str(neu),
            "agreeable": str(agr),
            "conscientious": str(con),
            "open": str(ope)
        })
        file = open(f"{self.output_path}/{user_id}.xml", "w")
        file.write(et.tostring(user, encoding="unicode"))

    def _set_baseline_model(self, test_data_set, pre_trained=True):
        """
        Sets the internal baseline model

        Parameters
        ---------
        test_data_set : datavengers.model.data.Data
            Object containing the oxford, profile, relational and liwc data sets
        pre_trained : bool, optional
            Flag which determines if a pre-trained model is loaded
        """
        if pre_trained:
            self._baseline_model = BaselineModel(self.train_data, test_data_set, self.load_path)
            model_path = f"{self.load_path}/baseline.model"
            self._baseline_model.load(model_path)

    def _set_gender_model(self, pre_trained=True):
        """
        Sets the internal gender prediction model

        Parameters
        ---------
        pre_trained : bool, optional
            Flag which determines if a pre-trained model is loaded
        """
        if pre_trained:
            self.gender_prediction_model = Gender()
            self.gender_prediction_model.train(self.train_data)

    def _set_personality_model(self, pre_trained=True):
        """
        Sets the internal personality prediction model

        Parameters
        ---------
        pre_trained : bool, optional
            Flag which determines if a pre-trained model is loaded
        """
        if pre_trained:
            self.personality_prediction_model = Personality()
            self.personality_prediction_model.load_model()

    def _set_age_model(self, pre_trained=True):
        """
        Sets the internal age prediction model

        Parameters
        ---------
        pre_trained : bool, optional
            Flag which determines if a pre-trained model is loaded
        """
        if pre_trained:
            self.age_prediction_model = AgePredictor()
            self.age_prediction_model.load_model()

    def get_predictions(self, test_data):
        """
        Obtains gender, personality and age predictions for test_data and generates the prediction files per user

        Parameters
        ---------
        test_data : datavengers.model.data.Data
            Object containing the oxford, profile, relational and liwc data sets
        """
        test_profiles = test_data.get_profiles()
        self._set_gender_model()
        self._set_personality_model()
        self._set_age_model()
        perso_preds = self.personality_prediction_model.predict(test_data)
        age_preds = self.age_prediction_model.predict(test_data)
        gender_preds = self.gender_prediction_model.predict(test_data)

        for uid, g, pp in zip(test_profiles["userid"], gender_preds, perso_preds):
            ope, neu, ext, agr, con = pp[0], pp[1], pp[2], pp[3], pp[4]
            self._write_to_xml(uid, age_preds[uid], g, ext, neu, agr, con, ope)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse command line arguments")
    parser.add_argument('-i', dest="test_data_path", help="absolute path to a test dataset with new instances", required=True)
    parser.add_argument('-o', dest="output_folder", help="absolute path to an empty output directory", required=True)
    parser.add_argument('-t', dest="train_data_path", help="absolute path to a training dataset")
    parser.add_argument('-l', dest="load_path", help="absolute path to a trained model dump file")
    args = parser.parse_args()
    test_data_path = args.test_data_path
    load_path = args.load_path if args.load_path else "./datavengers/persistence/"
    td = Data(f"{test_data_path}/Text/liwc.csv",
                          f"{test_data_path}/Text/nrc.csv",
                          f"{test_data_path}/Relation/Relation.csv",
                          f"{test_data_path}/Image/oxford.csv",
                          f"{test_data_path}/Profile/Profile.csv")
    master = Master(args.train_data_path, load_path, args.output_folder)
    master.get_predictions(td)
