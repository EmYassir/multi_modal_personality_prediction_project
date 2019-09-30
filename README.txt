First, execute the env_config script in order to prepare the environment for the python script execution:

- source env_config.sh

Then, execute the the ift6758.py script. Disclaimer: the -t argument, which is the path to a training set, is required for this first submission, given the simplicity of the baseline training and prediction

usage: python ift6758.py [-h] -i TEST_DATA_PATH -o OUTPUT_FOLDER -t TRAIN_DATA_PATH

  -h, --help          show this help message and exit
  -i TEST_DATA_PATH   absolute path to a test dataset with new instances
  -o OUTPUT_FOLDER    absolute path to an empty output directory
  -t TRAIN_DATA_PATH  absolute path to a training dataset

