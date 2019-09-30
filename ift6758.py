import argparse
import os
from datavengers.model.manager.model_manager import ModelManager

parser = argparse.ArgumentParser(description="Parse command line arguments")
parser.add_argument('-i', dest="test_data_path", help="absolute path to a test dataset with new instances")
parser.add_argument('-o', dest="output_folder", help="absolute path to an empty output directory")
args = parser.parse_args()
train_data_path_default = os.getcwd() + "/"
print(train_data_path_default)
manager = ModelManager(train_data_path_default, args.test_data_path, args.output_folder)
manager.train_all()
manager.predict_all()
