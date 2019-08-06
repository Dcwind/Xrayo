from __future__ import division, print_function
# coding=utf-8
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Keras
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template, jsonify
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# You can also use pretrained model from Keras
# Check https://keras.io/applications/

model = load_model("./models/3-conv-128-layer-dense-1-out-2-softmax-categorical-cross-2-CNN.model")

print('Running on http://localhost:5000')

def get_file_path_and_save(request):
    # Get the file from post request
    f = request.files['file']

    # Save the file to ./uploads
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(
        basepath, 'uploads', secure_filename(f.filename))
    f.save(file_path)
    return file_path

def prepare(filepath):
    img_size = 150  
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)/255  
    img_resize = cv2.resize(img, (img_size, img_size))  
    return np.array(img_resize).reshape(-1, img_size, img_size, 1) 
    

def c_scoref(vals):
    print(vals)
    #confidence score
    # [Dog, Cat]    

    if vals[1] > vals[0]:
        prediction = "PNEUMONIA"
        c_score = vals[1]/sum(vals) * 100
    else:
        prediction = "NORMAL"
        c_score = vals[0]/sum(vals) * 100
    print("Prediction:", prediction)
    print("Confidence score:",str(round(c_score,2)) + "%\n")
    return (prediction, round(c_score,2))


@app.route('/predictVGG16', methods=['GET', 'POST'])
def predictVGG16():
    if request.method == 'POST':
        file_path = get_file_path_and_save(request)

        prediction=model.predict(prepare(file_path))
        v = prediction[0]
        result = c_scoref(v)

        # decode the results into a list of tuples (class, description, probability)
        pred = str(result[0])
        c_val = str(result[1])

        print(pred, c_val)

        res = [pred, c_val]

        return jsonify(res)
    return None

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


if __name__ == '__main__':
    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
