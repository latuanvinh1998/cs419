from flask import Flask, jsonify
from flask_restful import Api
from flask_cors import CORS
import os, io
import numpy as np
import faiss
from flask import request
import base64
from PIL import Image
import torch
from torchvision import transforms
from torch import nn

from efficientnet_pytorch import EfficientNet
import cv2

app = Flask(__name__)
api = Api(app)
CORS(app)

tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])

def init_model():
	model = EfficientNet.from_pretrained('efficientnet-b7')
	model.eval()
	index = faiss.read_index("image_retrieval.bin")
	image_list = open('index.txt', 'r').read().split('\n')

	return model, index, image_list

def extract(model, image):

	with torch.no_grad():
		features = model.extract_features(image)
		features = nn.AdaptiveAvgPool2d(1)(features)
		features = torch.squeeze(features, -1)
		features = torch.squeeze(features, -1).cpu().detach().numpy()

	return features

model = None
index = None
image_list = None


@app.route('/', methods=['POST'])

def image_retrieval():
	return_string = ''
	request_data = request.get_json()
	img_str = request_data.get('file')
	img_str = img_str.split(',')
	img_data = base64.b64decode(str(img_str[1]))

	# img = request.files['data']
	# img = Image.open(img.stream)
	# img = img.save('images/file_upload.jpg')

	with open('images/file_upload.jpg', "wb") as fh:
		fh.write(img_data)
	img_data = tfms(Image.open('images/file_upload.jpg')).unsqueeze(0)
	# img_data = tfms(Image.open('img.jpg')).unsqueeze(0)
	features = extract(model, img_data)
	D, I = index.search(features, 10)
	I = I[0].tolist()
	for id in range(9):
		return_string += image_list[I[id]] + '\n'
	return_string += image_list[I[9]]

	result = {'list': return_string}
	return jsonify(result)

if __name__ == '__main__':
	#serve(app, host="127.0.0.1", port=5000)
	model, index, image_list = init_model()
	print('Finish init model.')
	app.debug = True
	app.run(host='0.0.0.0',port=8080)