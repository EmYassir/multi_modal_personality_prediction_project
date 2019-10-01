import argparse
from datavengers.model.manager.baseline_model import BaselineModel

parser = argparse.ArgumentParser(description="Parse command line arguments")
parser.add_argument('-i', dest="test_data_path", help="absolute path to a test dataset with new instances", required=True)
parser.add_argument('-o', dest="output_folder", help="absolute path to an empty output directory", required=True)
parser.add_argument('-t', dest="train_data_path", help="absolute path to a training dataset")
parser.add_argument('-l', dest="load_path", help="absolute path to a trained model dump file")
args = parser.parse_args()
model = BaselineModel(args.train_data_path, args.test_data_path, args.output_folder)
load_path = args.load_path if args.load_path else "baseline.dump"
print(f"Loading pre-trained baseline model")
model.load(load_path)
model.compute_predictions()
print(f"Output files generated in {args.output_folder}")
