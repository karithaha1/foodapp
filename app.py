from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from PIL import ImageOps, Image, ImageDraw, ImageFont
import os


class_labels = ['arphrong', 'kanomjeen', 'loba', 'mheehokkien', 'mheehoon', 'mhuhong', 'nampikkung', 'oaew', 'otao', 'popia']

model = load_model(os.path.join('static/model'))


app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = os.path.join('static/img/',(imagefile.filename))
    imagefile.save(image_path)
    img = image.load_img(image_path, target_size=(200, 200))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    val = model.predict(images)
    index = np.argmax(val)
    result = (class_labels[index])
    return render_template('result.html', prediction=result, image_path=image_path)

@app.route('/', methods=['GET', 'POST'])
def back():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(port=3000, debug=True)
