import pandas as pd
import numpy as np
import pickle
import streamlit as st
from PIL import Image,ImageOps
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model

def welcome():
    return 'welcome all'
# pickle_in = open('D:\MLA\model.pkl', 'rb')
# model = pickle.load(pickle_in)
model=load_model("mla.h5")
st.set_option('deprecation.showfileUploaderEncoding', False)
html_temp = """
<div class="bg">
    <div style ="background-color:blue;padding:13px">
    <h1 style ="color:black;text-align:center;"> wScan: An automated framework for multiclass classification of weather using Transfer learning apporach </h1>
    </div>
    </div>
    <style>
        .bg {
            background-image: url("D:\MLA\istockphoto-477110708-170667a");
            background-size: cover;
        }
        </style>
    """
st.markdown(html_temp, unsafe_allow_html=True)
file = st.file_uploader("Please upload an image", type=["jpg", "png"])
import cv2
from PIL import Image, ImageOps
import numpy as np


# this line allows us to display the front end aspects we have
# defined in the above code



def import_and_predict(image_data, model):
    size = (180, 180)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.

    img_reshape = img[np.newaxis, ...]

    prediction = model.predict(img_reshape)

    return prediction
image = Image.open(file)
st.image(image, use_column_width=True)
predictions = import_and_predict(image, model)
class_names=['Cloudy', 'Rain', 'Shine', 'Sunrise']
#score = tf.nn.softmax(predictions[0])
#st.write(prediction)
#st.write(score)

if st.button("Predict"):
    output_class = class_names[np.argmax(predictions)]
    st.success('The image is of {}'.format(output_class))
