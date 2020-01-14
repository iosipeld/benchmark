import tensorflow as tf

import time
import numpy as np
import cv2
import base64
import requests
import json


img = cv2.imread('./data/people.jpg')
img = cv2.resize(img,(416,416))

if img is None:
    print('no image')

img_in = tf.expand_dims(img, 0)
img_in = transform_images(img_in, 3)

image = np.expand_dims(img, axis=0)
payload = {"instances": image.tolist()}

for i in range(30):
    t1 = time.time()
    json_response = requests.post("http://localhost:8501/v1/models/yolov3:predict", json=payload)
    print((time.time() - t1) * 1000)

print(json_response.text)

# Extract text from JSON
response = json.loads(json_response.text)

# Interpret bitstring output
response_string = response["predictions"][0]["b64"]
print("Base64 encoded string: " + response_string[:10] + " ... " + response_string[-10:])

# Decode bitstring
encoded_response_string = response_string.encode("utf-8")
response_image = base64.b64decode(encoded_response_string)
print("Raw bitstring: " + str(response_image[:10]) + " ... " + str(response_image[-10:]))

# Save inferred image
with open("output/people.jpg", "wb") as output_file:
    output_file.write(response_image)
