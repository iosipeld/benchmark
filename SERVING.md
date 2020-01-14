# Serving

`docker run --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -it -p 8500:8500 -v $PWD/serving/yolov3:/yolo tensorflow/serving:latest-devel`
`tensorflow_model_server --model_base_path=/yolo --model_name=yolov3 --port=8500`