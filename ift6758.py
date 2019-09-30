import argparse
from datavengers.model.manager.model_manager import ModelManager

parser = argparse.ArgumentParser(description="Parse command line arguments")
parser.add_argument('-i', dest="test_data_path", help="absolute path to a test dataset with new instances", required=True)
parser.add_argument('-o', dest="output_folder", help="absolute path to an empty output directory", required=True)
parser.add_argument('-t', dest="train_data_path", help="absolute path to a training dataset", required=True)
args = parser.parse_args()
manager = ModelManager(args.train_data_path, args.test_data_path, args.output_folder)
manager.train_all()
manager.predict_all()
