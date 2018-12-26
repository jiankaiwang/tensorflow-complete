#/bin/bash

LOGDIR=/notebooks/object_detection/model
tensorboard --logdir=${LOGDIR} > /dev/null 2>&1 & 
echo $! > /tmp/tfb.run