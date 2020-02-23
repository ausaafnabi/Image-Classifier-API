import base64
import json
from io import BytesIO

import numpy as np
import requests
from flask import Flask,requests,jsonify
from keras.applications import inception_v3
from keras.preprocessing import image

app = Flask(__name__)

app.route('/imageclassifier/predict',method=['POST'])
def image_classifier():
	img = image.img_to_array(image.load_img(BytesIO(base64.b64decode(request.form['b64'])), target_size=(224,224)))/225.

	img =img.astype('float16')

	payload = {
		"instances":[{'input_image': img.tolist()}]
	}

	r = requests.post('http://localhost:9800/v1/models/ImageClassifier:predict',json =payload)

	pred = json.loads(r.content.decode('utf-8'))

	return jsonify(inception_v3.decode_predictions(np.array(pred['predictions']))[0])

