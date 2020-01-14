# Tensorboard
All script must be executed from the root of this project.

## import_pb_to_tensorboard.py

Generate: `python import_pb_to_tensorboard.py --model_dir $PWD/models/darknet-yolov3/1 --tag_set serve --log_dir /tmp/logs/darknet-yolov3/1`
Run: `tensorboard --logdir=/tmp/logs/darknet-yolov3/1`

## import_pbtxt_to_tensorboard.py

Generate: `python import_pbtxt_to_tensorboard.py --model_dir $PWD/optimized-graphs/darknet-yolov3/after_MetaOptimizer_140575076444688.pbtxt --tag_set serve --log_dir /tmp/logs/darknet-yolov3/2`
Run: `tensorboard --logdir=/tmp/logs/darknet-yolov3/2`
