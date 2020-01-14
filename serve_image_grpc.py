import tensorflow as tf
import time
import numpy
import cv2
import grpc
import argparse
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

parser = argparse.ArgumentParser(description='gRPC image request flags.')
parser.add_argument('--host', default='0.0.0.0', help='TFServing model server')
parser.add_argument('--port', default='8500', help='TFServing model server gRPC endpoint port')
parser.add_argument('--model', default='darknet-yolov3', help='Model name used for inference')
parser.add_argument('--image', default='./data/street.jpg', help='Path to a image file')
parser.add_argument('--size', default='416', help='Size of the video frame')
FLAGS = parser.parse_args()

RPC_TIMEOUT = 30.0


def main():
    start = time.time()
    # Create request for a model server
    request = predict_pb2.PredictRequest()
    request.model_spec.name = FLAGS.model
    request.model_spec.signature_name = 'serving_default'

    # Create prediction service stub
    channel = grpc.insecure_channel("{}:{}".format(FLAGS.host, FLAGS.port))
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    # Read image into numpy array
    image = cv2.imread(FLAGS.image).astype(numpy.float32)
    if image is None:
        print('Invalid image provided: {}'.format(FLAGS.image))

    image = cv2.resize(image, (int(FLAGS.size), int(FLAGS.size)))

    tensor = tf.make_tensor_proto(image, shape=[1] + list(image.shape))
    request.inputs['input_1'].CopyFrom(tensor)
    response = stub.Predict(request, RPC_TIMEOUT)

    print('Execution time: {} ms'.format((time.time() - start) * 1000))
    #print(response)


if __name__ == '__main__':
    main()
