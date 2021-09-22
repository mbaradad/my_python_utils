#!/usr/bin/env bash
export HOME=/data/vision/torralba/movies_sfm/home
cd $HOME

#add my_libs
export PATH="$HOME/programs/bin:$PATH"
# added by Anaconda2 installer
if [[ $PATH = *$HOME/anaconda* ]]
then
    echo "Anaconda path already present, wont be added again!"
else
    export PATH="$HOME/anaconda3/bin:$PATH"
fi

source activate default_env37
source $HOME/.bash_commands/all
cd $HOME/no_training_cnn/contrastive_image_models;
export PYTHONPATH=.
export COMMON_ARGS="--generate True --parallel True --impose_wmm True --correlate_channels True"
python image_generation/statistical_image_models/generate_datasets.py $COMMON_ARGS --impose_spectrum False --impose_color_model False --wmm-model 0
python image_generation/statistical_image_models/generate_datasets.py $COMMON_ARGS --impose_spectrum False --impose_color_model True --wmm-model 0
python image_generation/statistical_image_models/generate_datasets.py $COMMON_ARGS --impose_spectrum True --impose_color_model False --wmm-model 0
python image_generation/statistical_image_models/generate_datasets.py $COMMON_ARGS --impose_spectrum True --impose_color_model True --wmm-model 0

python image_generation/statistical_image_models/generate_datasets.py $COMMON_ARGS --impose_spectrum False --impose_color_model False --wmm-model 1
python image_generation/statistical_image_models/generate_datasets.py $COMMON_ARGS --impose_spectrum False --impose_color_model True --wmm-model 1
python image_generation/statistical_image_models/generate_datasets.py $COMMON_ARGS --impose_spectrum True --impose_color_model False --wmm-model 1
python image_generation/statistical_image_models/generate_datasets.py $COMMON_ARGS --impose_spectrum True --impose_color_model True --wmm-model 1

python image_generation/statistical_image_models/generate_datasets.py $COMMON_ARGS --impose_spectrum False --impose_color_model False --wmm-model 2 --delta_p_range_0 0 --delta_p_range_1 0.95
python image_generation/statistical_image_models/generate_datasets.py $COMMON_ARGS --impose_spectrum False --impose_color_model True --wmm-model 2 --delta_p_range_0 0 --delta_p_range_1 0.95
python image_generation/statistical_image_models/generate_datasets.py $COMMON_ARGS --impose_spectrum True --impose_color_model False --wmm-model 2 --delta_p_range_0 0 --delta_p_range_1 0.95
python image_generation/statistical_image_models/generate_datasets.py $COMMON_ARGS --impose_spectrum True --impose_color_model True --wmm-model 2 --delta_p_range_0 0 --delta_p_range_1 0.95

python image_generation/statistical_image_models/generate_datasets.py $COMMON_ARGS --impose_spectrum False --impose_color_model False --wmm-model 2 --delta_p_range_0 0.5 --delta_p_range_1 0.95
python image_generation/statistical_image_models/generate_datasets.py $COMMON_ARGS --impose_spectrum False --impose_color_model True --wmm-model 2 --delta_p_range_0 0.5 --delta_p_range_1 0.95
python image_generation/statistical_image_models/generate_datasets.py $COMMON_ARGS --impose_spectrum True --impose_color_model False --wmm-model 2 --delta_p_range_0 0.5 --delta_p_range_1 0.95
python image_generation/statistical_image_models/generate_datasets.py $COMMON_ARGS --impose_spectrum True --impose_color_model True --wmm-model 2 --delta_p_range_0 0.5 --delta_p_range_1 0.95

python image_generation/statistical_image_models/generate_datasets.py $COMMON_ARGS --impose_spectrum False --impose_color_model False --wmm-model 2 --delta_p_range_0 0.9 --delta_p_range_1 0.95
python image_generation/statistical_image_models/generate_datasets.py $COMMON_ARGS --impose_spectrum False --impose_color_model True --wmm-model 2 --delta_p_range_0 0.9 --delta_p_range_1 0.95
python image_generation/statistical_image_models/generate_datasets.py $COMMON_ARGS --impose_spectrum True --impose_color_model False --wmm-model 2 --delta_p_range_0 0.9 --delta_p_range_1 0.95
python image_generation/statistical_image_models/generate_datasets.py $COMMON_ARGS --impose_spectrum True --impose_color_model True --wmm-model 2 --delta_p_range_0 0.9 --delta_p_range_1 0.95
