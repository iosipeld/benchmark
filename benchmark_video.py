import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
# os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '1'

import cv2
import argparse
import numpy
import tensorflow
import grpc
import time
import datetime
import threading
import sys
from hdrh.histogram import HdrHistogram
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

parser = argparse.ArgumentParser(description='Video benchmark flags.')
parser.add_argument('--host', default='0.0.0.0', help='TFServing model server')
parser.add_argument('--port', default='8500', help='TFServing model server gRPC endpoint port')
parser.add_argument('--model', default='darknet-yolov3', help='Model name used for inference')
parser.add_argument('--video', default='./data/busy-road.mp4', help='Path to a video file or camera')
parser.add_argument('--size', default='416', help='Size of the video frame')
parser.add_argument('--concurrency', default=1)
FLAGS = parser.parse_args()

RPC_TIMEOUT = 30.0


class _State:
    """Holds a state of the execution."""

    def __init__(self, total, concurrency):
        self._total = total
        self._concurrency = concurrency
        self._active = 0
        self._done = 0
        self._error = 0
        self._condition = threading.Condition()
        self._histogram = HdrHistogram(1, 300000000000, 4)

        self._minLatency = sys.maxsize
        self._maxLatency = int('-inf')

    def inc_error(self):
        with self._condition:
            self._error += 1

    def inc_total(self):
        with self._condition:
            self._total += 1
            self._condition.notify()

    def inc_done(self):
        with self._condition:
            self._done += 1
            self._condition.notify()

    def dec_active(self):
        with self._condition:
            self._active -= 1
            self._condition.notify()

    def record_histogram(self, microseconds):
        self._histogram.record_value(microseconds)
        self._minLatency = min(self._minLatency, microseconds)
        self._maxLatency = max(self._maxLatency, microseconds)

    def get_error_rate(self):
        with self._condition:
            while self._done != self._total:
                self._condition.wait()
            return self._error / float(self._total)

    def get_execution_time_jitter(self):
        return self._maxLatency - self._minLatency

    def get_percentile(self, percentile):
        return self._histogram.get_value_at_percentile(percentile)

    def throttle(self):
        with self._condition:
            while self._active == self._concurrency:
                self._condition.wait()
            self._active += 1

    def print_histogram(self):
        for item in self._histogram.get_recorded_iterator():
            print('count=%d percentile=%f' %
                  item.value_iterated_to,
                  item.count_added_in_this_iter_step,
                  item.percentile)

    def save_measurements(self):
        self._histogram.output_percentile_distribution(open('measurements/{}.txt'.format(int(time.time())), 'wb'), 1000)


def _create_rpc_callback(state):
    """Creates RPC callback function.
    Args:
      state: Overall state of the benchmark.
    Returns:
      The callback function.
    """
    start = datetime.datetime.now()

    def _callback(result_future):
        """Callback function.
        Adds latency to the histogram.
        Args:
          result_future: Result future of the RPC.
        """
        exception = result_future.exception()
        if exception:
            state.inc_error()
            print(exception)
        else:
            state.record_histogram((datetime.datetime.now() - start).total_seconds() * 1000)
            state.inc_done()
            state.dec_active()

    return _callback


def main():
    # Create request for a model server
    request = predict_pb2.PredictRequest()
    request.model_spec.name = FLAGS.model
    request.model_spec.signature_name = 'serving_default'

    # Create prediction service stub
    channel = grpc.insecure_channel("{}:{}".format(FLAGS.host, FLAGS.port))
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    # Concurrency defines how many max requests are in a queue for processing
    state = _State(0, int(FLAGS.concurrency))

    # Read webcam or video file
    try:
        video = cv2.VideoCapture(int(FLAGS.video))
        # video.set(cv2.CAP_PROP_FPS, 1) Does not work as expected, device driver must be capable to set it (for Linux)
    except:
        video = cv2.VideoCapture(FLAGS.video)

    while True:
        success, raw_frame = video.read()

        if success is False:
            print("Video read failed!")
            break

        state.inc_total()

        # Convert frame to Tensor
        float_frame = raw_frame.astype(numpy.float32)

        image = cv2.resize(float_frame, (int(FLAGS.size), int(FLAGS.size)))
        tensor = tensorflow.make_tensor_proto(image, shape=[1] + list(image.shape))

        # Prepare model input
        request.inputs['input_1'].CopyFrom(tensor)

        state.throttle()
        print('predict')
        # Predict
        predict = stub.Predict.future(request)
        predict.add_done_callback(_create_rpc_callback(state))

    print("Error ration: {}".format(state.get_error_rate()))
    print("Mean: {}".format(state.get_percentile(95)))
    print("Jitter: {}".format(state.get_execution_time_jitter()))

    state.save_measurements()
    # Tegra stats parsing removed as it was heavily inspired by
    # https://github.com/rbonghi/jetson_stats/blob/master/jtop/core/tegra_parse.py


if __name__ == '__main__':
    main()
