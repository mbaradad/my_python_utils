#!/bin/bash

LOCAL_SSD1=/mnt/localssd1
if [ -d $LOCAL_SSD1 ]
then
    SSD_DATASETS_PATH=$LOCAL_SSD1/datasets
    sudo mkdir -p $SSD_DATASETS_PATH
    sudo chown mbaradad:mbaradad $SSD_DATASETS_PATH
    sudo chmod 777 $SSD_DATASETS_PATH
    yes | rsync -azPL visiongpu49:/mnt/localssd1/datasets/imagenet100 $SSD_DATASETS_PATH
fi