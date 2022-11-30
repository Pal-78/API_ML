#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import flask
import io
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, jsonify, request


# In[2]:


# import trained model
model = tf.keras.models.load_model('api_model.h5')


# In[3]:


def prepare_image(img):
    """load, resize and format picture for model"""
    img = Image.open(io.BytesIO(img))
    img = img.resize((224, 224))
    img = np.array(img)
    img = np.expand_dims(img, 0)
    return img


# In[4]:


app = Flask(__name__)


# In[5]:


@app.route('/predict', methods=['POST'])
def infer_image():
    # what is send must comply specific format with 'image'
    if 'image' not in request.files:
        return "Please try again. The Image doesn't exist"
    # upload file
    file = request.files.get('image')
    # check file exist
    if not file:
        return
    # read picture
    img_bytes = file.read()
    # adapte picture to model expectations
    img = prepare_image(img_bytes)
    # predict
    y = model.predict(img)
    # create mask
    mask = np.argmax(y, axis=3)
    # result
    return jsonify(mask.tolist())

@app.route('/', methods=['GET'])
def index():
    return '<h1>Future Vision Transport API</h1>'


# In[6]:


if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:
