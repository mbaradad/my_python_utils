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
cd $HOME/no_training_cnn;
./scripts/datasets/fragments/tar_images_per_fragment.sh