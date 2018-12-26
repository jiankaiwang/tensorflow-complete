# Tensorflow For Object Dection API

This docker image is also designed for object detection API on Tensorflow. Main reference please refer to 

* https://github.com/tensorflow/models/tree/master/research/object_detection.

We provide both CPU-/GPU-based docker images. They are similar so the instructions below would use GPU-based image as the example.

## Quickstart

Here we guide you how to quickly start a training and inference.

### Starting a container

Make sure you have pull the image down first.

```sh
# pull gpu-based docker image
# instead you can 
# pull jiankaiwang/tensorflow-complete:1.11.0-py3
# to pull down a cpu-based docker image
docker pull jiankaiwang/tensorflow-complete:1.11.0-gpu-py3
```

And then start a container from the below command.

```sh
# here we recommand using nivida-docker to run the container with GPU supported
#
# -p: port forward
# -v: mount data volume (if you are going to train your own datasets)
# --device: mount you webcam, if you want to infer the real time video streaming
nvidia-docker run --rm -it --name tfc 
	-p 28888:8888 -p 26006:6006 -p 25000:5000 
	--device /dev/video0:/dev/video0 
	[-v <host path>:<container path>]
	tf-complete:1.11.0-gpu-py3
```

You now can surf the web (http://localhost:28888/) to start a new notebook. 

Further, you can access to the container via the below command. 

```sh
# access to a running container
docker exec -it tfc /bin/bash
```

### Training

1. Prepare data in tfrecord format. Replace the `train.tfrecord`, `val.tfrecord` and `label_map.pbtxt` on the folder `data`. (**We recommend that newcomers prepare dataset with the same names.**)
    * You can download from web via `curl` or `wget` commands.
    * You can mount the volume via `-v <host>:/notebooks/object_detection/data`.
2. Execute the bash `bash /notebooks/object_detection/start_object_detection.sh`.
    * After the training, you can find the frozen model (.pb) on `/notebooks/object_detection/model`.

### Tensorboard 

1. You can run the script `bash /notebooks/object_detection/start_tensorboard.sh` to minitor the training progress on `localhost:26006`.
2. You can simplt stop the monitoring via `bash /notebooks/object_detection/stop_tensorboard.sh`.

### Inference

1. You can start from jupyter notebook. [**Note: If you are going to run on pretrained model, make sure you have changed `your_own_model` to `False` on the jupyter notebook**.]
    * You can refer to the notebook`1_object_detection_tutorial_modified_from_official_repo.ipynb` modified from the official repository.
    * You can refer to the notebook `2_object_detection_less_requirement.ipynb` to inference the object detection in a more simpler way.
2. You can also inference a video from the script `Inference_SSD_from_realtime_streaming.py`. This is the basic demo for object detection from video streaming.
3. You can start object detection from realtime video streaming and also monitor the result from the web. (It might solve the problem that container can't access screen.)
    * Run the command `bash /notebooks/object_detection/start_webcam.sh` to start inference from webcam and to monitor the result from web (`http://IP:5000`).
    * Run the command `bash /notebooks/object_detection/stop_webcam.sh` to stop inference.



## Advanced

The below is detailed about how to configurate the pipeline from the scratch.

### Data Preparation

Please prepare train and eval data, both the `tfrecord` and `label` data are necessary. The below is the simple guide for preparing your own dataset. However, you can quick run the object detection training via small flower datasets located on `object_detection/data`.



The schema in tfrecord is as below:

```python
def create_cat_tf_example(encoded_cat_image_data):
   """Creates a tf.Example proto from sample cat image.

  Args:
    encoded_cat_image_data: The jpg encoded data of the cat image.

  Returns:
    example: The created tf.Example.
  """

  height = 1032.0
  width = 1200.0
  filename = 'example_cat.jpg'
  image_format = b'jpg'

  xmins = [322.0 / 1200.0]
  xmaxs = [1062.0 / 1200.0]
  ymins = [174.0 / 1032.0]
  ymaxs = [761.0 / 1032.0]
  classes_text = ['Cat']
  classes = [1]

  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_example
```

More information about how to create tfrecord please refer to:
* https://github.com/jiankaiwang/TF_ObjectDetection_Flow
* https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md

The content in label is as below:

```text
item {
  id: 1
  name: 'class1'
}
item {
  id: 2
  name: 'class2'
}
item {
  id: 3
  name: 'class3'
}

```



### Configurating the Pipeline

In this image, we demostrate how to train the dataset on `ssd_mobilenet_v1_coco` model. More pretrained models you can selected please refer to https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md.



If you want to train on your own dataset, you have to edit the pipeline. You can configurate the pipeline located on `object_detection/ssd_mobilenet_v1_coco_2018_01_28_docker/pipeline.config`. The whole resource is downloaded from the official github repository and modified to fit this image. Make sure you have at least changed labeling `PATH_TO_BE_CONFIGURED` on the pipeline, and `num_classes`. The content of QuickStart demo is like below:

```text
model {
  ssd {
    num_classes: 6
    ...
    (here is hyperparameters for training)
    ...
  }
}
train_config {
  batch_size: 24
  ...
  (here is the augmentation option)
  ...
  fine_tune_checkpoint: "/notebooks/object_detection/ssd_mobilenet_v1_coco_2018_01_28_docker/model.ckpt"
  from_detection_checkpoint: true
  num_steps: 200000
}
train_input_reader {
  label_map_path: "/notebooks/object_detection/datasets/label_map.pbtxt"
  tf_record_input_reader {
    input_path: "/notebooks/object_detection/datasets/train.record"
  }
}
eval_config {
  num_examples: 8000
  max_evals: 10
  use_moving_averages: false
}
eval_input_reader {
  label_map_path: "/notebooks/object_detection/datasets/label_map.pbtxt"
  shuffle: false
  num_readers: 1
  tf_record_input_reader {
    input_path: "/notebooks/object_detection/datasets/val.record"
  }
}
```



### Starting a Training and Exporting a Frozen Model

You can edit the `start_object_detection.sh` to package all requirements for training.

```sh
#!/bin/bash

export PYTHONPATH=$PYTHONPATH:/notebooks/models/research:/notebooks/models/research/slim

# retrained based on SSD_MobileNet_V1
HOMEDIR=/notebooks/object_detection
PIPELINE_CONFIG_PATH=$HOMEDIR/ssd_mobilenet_v1_coco_2018_01_28_docker/pipeline.config
CUS_PIPELINE_CONFIG_PATH=${HOMEDIR}/data/pipeline.config
MODEL_DIR=$HOMEDIR/model
NUM_TRAIN_STEPS=10000
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
RUNDIR=/notebooks/models/research

# customize pipeline.config
python ${HOMEDIR}/helpers/set_training_configuration.py \
    --you_own True \
    --label_map ${HOMEDIR}/data/label_map.pbtxt \
    --pipeline ${PIPELINE_CONFIG_PATH} \
    --pipelineoutput ${CUS_PIPELINE_CONFIG_PATH}
if [ $? != "0" ]; then
    echo "Training configuration was not generated. Training was stopping."
    exit 1
fi

# retrain a new object detection model
cd ${RUNDIR}
python object_detection/model_main.py \
    --pipeline_config_path=${CUS_PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=${SAMPLE_1_OF_N_EVAL_EXAMPLES} \
    --alsologtostderr

# copy the label file from pipeline.config
label_path=`python ${HOMEDIR}/helpers/parser.py --method labelfile --pipeline ${PIPELINE_CONFIG_PATH}`
cp ${label_path} ${MODEL_DIR}/label_map.pbtxt

# export frozen model
bash ${HOMEDIR}/helpers/export_object_detection_model.sh

exit 0
```

After packaging all requirements, you can start a training via `bash start_object_detection.sh`.



### Object detection via real time video streaming

The video streaming over flask is referred from <https://github.com/miguelgrinberg/flask-video-streaming>, the `Montion JPG`idea. In this docker, we use this idea to implement object detection on real-time video streaming. 

The simple realtime video streaming can refer to the script `object_detection/realtime_inference.py`.





