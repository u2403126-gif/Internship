import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model('rice_cnn_model.h5')

classes = [
    'Arborio',
    'Basmati',
    'Ipsala',
    'Jasmine',
    'Karacadag'
]

st.title("Rice Image Classifier")

uploaded_file = st.file_uploader("Upload Rice Image", type=['jpg','png','jpeg'])

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image)

    image = image.resize((128,128))

    image = np.array(image)/255.0

    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)

    result = classes[np.argmax(prediction)]

    st.success(result)