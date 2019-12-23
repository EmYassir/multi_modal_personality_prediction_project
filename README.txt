Firstly, it is required to create the conda environment. Please, execute manually the following commands:

    module load anaconda/3
    conda deactivate
    conda env remove --name datavengers
    yes | conda create -n datavengers
    conda activate datavengers
    yes | conda install numpy
    yes | conda install pandas
    yes | conda install seaborn
    yes | conda install scikit-learn
    yes | conda install tensorflow
    yes | conda install -c conda-forge keras
    export KERAS_BACKEND=tensorflow

    (We had to do it this way now because conda stopped working in subshells and our source_env.sh script is not
    working correctly anymore)

Afterwards, to obtain the predictions, the ift6758.py file may be executed with the following options:

    python ift6758.py $OPTIONS

   Required:
      -i TEST_DATA_PATH   absolute path to a test dataset with new instances
      -o OUTPUT_FOLDER    absolute path to an empty output directory

   Optional:
      -t TRAIN_DATA_PATH  absolute path to a training dataset
      -l LOAD_PATH        absolute path to a trained model dump file

