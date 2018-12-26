# Tensorflow with Models



This docker image is to create a complete executing environment running on tensorflow. Further, in the container, you can simply run different models/tasks, e.g. object detection, word embedding, etc, model details please refer to https://github.com/tensorflow/models. 



Second, this docker image also provides the quickstart of models, the practice content listed below. You can pull docker images from docker hub https://hub.docker.com/u/jiankaiwang/.



## Content



There are four domains listed in the tensorflow/model as listed, the docker image would prepare different environment for different models. 

* official: The model in this folder is well-maintained, tested, and kept up with the latest Tensorflow API. The official recommands starting here.
* research: The model in this folder is implemented by researchers, but it is not officially supported or not available in latest release branches.
    * [object_detection quickstart](object_detection/)
* samples: Provides with code snippets and smaller models demostrating features of Tensorflow.
* tutorials