# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 19:54:03 2020

@author: Vijay Thangavel
"""

from __future__ import division, print_function

import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True

session = InteractiveSession(config=config)

from keras import backend as K
#from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


#Flask 
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)
MODEL_PATH = './model_inception.h5'

#tf.compat.v1.disable_eager_execution()
global graph
graph = tf.compat.v1.get_default_graph()

def model_predict(img_path):
    global graph
#    model = load_model(MODEL_PATH)
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = x/255
    x = np.expand_dims(x, axis=0)
    print(x)
    tf.keras.backend.clear_session()
    model = tf.keras.models.load_model(MODEL_PATH)
    model.compile()
    model.run_eagerly = True
#    model._make_predict_function() 
    try:
#        with graph.as_default():
        preds = model.predict(x)
        preds = np.argmax(preds, axis=1)
        
    except ValueError as ve:
        preds = "ERROR: " + str(ve)
    
#    print(preds)
    if preds==0:
        msg = "DISEASED cotton LEAF"
    elif preds==1:
        msg="DISEASED cotton PLANT"
    elif preds==2:
        msg="FRESH cotton LEAF"
    elif preds==3:
        msg="FRESH cotton PLANT"
    else:
        msg=str(preds)
         
    return msg


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        print('here 1')
        print(request.files)
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        
        preds = model_predict(file_path)
        return preds
    return None

if __name__=='__main__':
    app.run(port=5001, debug=True)