import argparse
import numpy as np
import time
import requests
import cv2
import json

parser = argparse.ArgumentParser(description='HTTP image request flags.')
parser.add_argument('--host', default='0.0.0.0', help='TFServing model server')
parser.add_argument('--port', default='8501', help='TFServing model server gRPC endpoint port')
parser.add_argument('--model', default='darknet-yolov3', help='Model name used for inference')
parser.add_argument('--image', default='./data/street.jpg', help='Path to a image file')
parser.add_argument('--size', default='416', help='Size of the video frame')
FLAGS = parser.parse_args()

RPC_TIMEOUT = 30.0


def main():
    start = time.time()

    # Read image and move to numpy array
    image = cv2.imread(FLAGS.image).astype(np.float32)

    if image is None:
        print('Invalid image provided: {}'.format(FLAGS.image))

    image = cv2.resize(image, (int(FLAGS.size), int(FLAGS.size)))
    image = np.expand_dims(image, axis=0)

    # Serialize payload
    payload = {"instances": image.tolist()}

    # Request
    json_response = requests.post(
        "http://{}:{}/v1/models/{}:predict".format(FLAGS.host, FLAGS.port, FLAGS.model),
        json=payload
    )

    # Extract response body from JSON
    response = json.loads(json_response.text)

    print('Execution time: {} ms'.format((time.time() - start) * 1000))
    #print(json_response.text)


if __name__ == '__main__':
    main()
