# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 17:19:31 2021

@author: SAMBA
"""


from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
import string
import numpy as np
from PIL import Image
import os
from pickle import dump, load
import numpy as np
from keras.applications.xception import Xception, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.layers.merge import add
from keras.models import Model, load_model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout
# small library for seeing the progress of loops.
from tqdm import tqdm_notebook as tqdm



app = Flask(__name__)
model = load(open('C:\\Users\\SAMBA\\Desktop\\Internship DS\\ImageCaptionGen\\model.h5', 'rb'))

@app.route("/")

def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def extract_features(filename, model):
    try:
        image = Image.open(filename)
    except:
        print("ERROR: Couldn't open image! Make sure the image path and extension is correct")
    image = image.resize((299,299))
    image = np.array(image)
    # for images that has 4 channels, we convert them into 3 channels
    if image.shape[2] == 4: 
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)
    image = image/127.5
    image = image - 1.0
    feature = model.predict(image)
    return feature
def word_for_id(integer, tokenizer):
  for word, index in tokenizer.word_index.items():
     if index == integer:
         return word
  return None
def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo,sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text

def predict():
    max_length = 32
    tokenizer = load(open("tokenizer.pkl","rb"))
    model = load_model('model.h5')
    xception_model = Xception(include_top=False, pooling="avg")
    img_path=request.files['img']
    photo = extract_features(img_path, xception_model)
    description = generate_desc(model, tokenizer, photo, max_length)
    return render_template('home.html',prediction_text=description[1:-1])
    
if __name__ == "__main__":
    app.run(debug=True)
