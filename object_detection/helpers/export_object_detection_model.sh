#!/bin/bash

export PYTHONPATH=$PYTHONPATH:/notebooks/models/research:/notebooks/models/research/slim

DATA_PATH=/notebooks/object_detection/data
PRETRAINED_PATH=/notebooks/object_detection/ssd_mobilenet_v1_coco_2018_01_28_docker
TRAINED_PATH=/notebooks/object_detection/model
HELPERS_DIR=/notebooks/object_detection/helpers

# get the latest checkpoint
latest=`python ${HELPERS_DIR}/parser.py --method lastestCheckpoint --ckpt ${TRAINED_PATH}`

# check whether pre-exported model exists
SAVED_MODEL_PATH=${TRAINED_PATH}/saved_model
if [ -d ${SAVED_MODEL_PATH} ]; then
while true; do
    read -p "Overwrite pre-exported model? [y/n] " yn
    case $yn in
        [Yy]* ) rm -rf ${SAVED_MODEL_PATH}; break;;
        [Nn]* ) printf "Exporting model stopped.\n"; exit;;
        * ) echo "Please answer yes or no.\n";;
    esac    
done        
fi

# export the frozen model
# trained_checkpoint_prefix: can be model.ckpt or model.ckpt-10000, etc.
cd /notebooks/models/research
python object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path ${PRETRAINED_PATH}/pipeline.config \
    --trained_checkpoint_prefix ${latest} \
    --output_directory ${TRAINED_PATH}

# copy label file
cp ${DATA_PATH}/label_map.pbtxt ${TRAINED_PATH}
