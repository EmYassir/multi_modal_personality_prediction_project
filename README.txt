Firstly, it is required to execute the env_config script in order to prepare the environment for the predictor.
The exact following command must be executed:

    source env_config.sh

Afterwards, to obtain the predictions, the ift6758.sh file may be executed with the following options:

    ./ift6758 $OPTIONS

   Required:
      -i TEST_DATA_PATH   absolute path to a test dataset with new instances
      -o OUTPUT_FOLDER    absolute path to an empty output directory

   Optional:
      -t TRAIN_DATA_PATH  absolute path to a training dataset
      -l LOAD_PATH        absolute path to a trained model dump file

