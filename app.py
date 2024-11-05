# import tensorflow as tf 
# from tensorflow import keras
# from tensorflow.keras.models import load_model
# import numpy as np
# import streamlit as st

# st.header('Identify')
# data_cat = ['apple',
#  'banana',
#  'beetroot',
#  'bell pepper',
#  'cabbage',
#  'capsicum',
#  'carrot',
#  'cauliflower',
#  'chilli pepper',
#  'corn',
#  'cucumber',
#  'eggplant',
#  'garlic',
#  'ginger',
#  'grapes',
#  'jalepeno',
#  'kiwi',
#  'lemon',
#  'lettuce',
#  'mango',
#  'onion',
#  'orange',
#  'paprika',
#  'pear',
#  'peas',
#  'sweetcorn',
#  'sweetpotato',
#  'tomato',
#  'turnip',
#  'watermelon']

# model = load_model('Image_classify.keras')

# img = st.text_input('Enter Image Name','download.jpeg')

# image = tf.keras.utils.load_img(img, target_size=(180, 180))
# img_arr = tf.keras.utils.array_to_img(image)
# img_bat=tf.expand_dims(img_arr,0)

# predict = model.predict(img_bat)

# score = tf.nn.softmax(predict)
# st.image(img, width = 22)
# st.write("Image Name: ", data_cat[np.argmax(score)])
# st.write("Accuracy: ", np.max(score) * 100)


import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import streamlit as st
from PIL import Image

data_cat = ['apple',
 'banana',
 'beetroot',
 'bell pepper',
 'cabbage',
 'capsicum',
 'carrot',
 'cauliflower',
 'chilli pepper',
 'corn',
 'cucumber',
 'eggplant',
 'garlic',
 'ginger',
 'grapes',
 'jalepeno',
 'kiwi',
 'lemon',
 'lettuce',
 'mango',
 'onion',
 'orange',
 'paprika',
 'pear',
 'peas',
 'sweetcorn',
 'sweetpotato',
 'tomato',
 'turnip',
 'watermelon']

st.set_page_config(page_title="Fruit & Vegetable Identifier", page_icon="🍎", layout="centered")

st.title("Identify Fruits & Vegetables 🍎🍌🥕")
st.subheader("Upload an image to identify the item")

model = load_model('Image_classify.keras')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    image = Image.open(uploaded_file)
    image = image.resize((180, 180))
    # img_array = np.array(image) / 255.0
    img_arr = tf.keras.utils.array_to_img(image)
    img_batch = tf.expand_dims(img_arr, axis=0)

    prediction = model.predict(img_batch)
    score = tf.nn.softmax(prediction[0])

    st.write("### Prediction")
    st.write(f"**Category**: {data_cat[np.argmax(score)]}")
    st.write(f"**Confidence**: {np.max(score) * 100:.2f}%")
else:
    st.info("Please upload an image to classify.")
