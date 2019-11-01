#!/bin/bash -i

# Load conda and create the datavengers env
module load anaconda/3
conda deactivate
conda env remove --name datavengers
yes | conda create -n datavengers
conda init bash
conda activate datavengers
yes | conda install numpy
yes | conda install pandas
yes | conda install scikit-learn
chmod +x ift6758
