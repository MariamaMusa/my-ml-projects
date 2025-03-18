import streamlit as st
import tensorflow
import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import joblib
import pandas as pd

model = tensorflow.keras.models.load_model("Fruit_image_Classifier.h5")
fruit_classes = ["Apple", "Banana", "Avocado", "Cherry", "Kiwi", "Mango", "Orange", "Pineapple", "Strawberry", "Watermelon"]


st.title("Fruit Image Classifier")
st.write("Heyy there, welcome!☺️")
st.write("I can tell you the name of your fruit if you submit a picture of it. Sounds interesting right? Give it a try!😌")
st.write(f"Remember that your fruit would have to be one of  {fruit_classes} for me to able to tell you what it is")
st.write("Now take a picture of your fruit using your camera or upload an existing image of it")
st.divider()


picture = st.camera_input("Point your fruit to the camera to capture it")
st.write("OR")
fruit_image = st.file_uploader("Upload an existing image of your fruit", type=["png", "jpg", "jpeg"])


if picture is not None:
    picture = Image.open(picture)
    picture = picture.convert("RGB")
    picture = picture.resize((222,222))
    picture = np.array(picture) / 255.0
    picture = np.expand_dims(picture, axis=0)


if fruit_image is not None:
    fruit_image = Image.open(fruit_image)
    fruit_image = fruit_image.convert("RGB")
    fruit_image = fruit_image.resize((222,222))
    fruit_image = np.array(fruit_image) / 255.0
    fruit_image = np.expand_dims(fruit_image, axis=0)
    st.image(fruit_image, caption="This is a resized image of the fruit you uploaded", use_container_width=False)




if st.button("Predict the Fruit"):
    if picture is None and fruit_image is None:
        st.warning("Ooops!🫠 You need to submit a picture of your fruit to continue!")
    elif picture is not None and fruit_image is None:
        prediction = model.predict(picture)
        st.write("The probability that your fruit is any of the 10 fruits is shown below")

        fruit = []
        values = []
        for i in range(len(fruit_classes)):
            fruit.append(fruit_classes[i])
            values.append(float(prediction[0][i]))

            if fruit == fruit_classes:
                us = pd.DataFrame({"FRUIT": fruit, 
                                "MY PROBABILITY PREDICTION": values})
                
                st.dataframe(us, hide_index=True)


        predicted_class_index = np.argmax(prediction)
        predicted_class_name = fruit_classes[predicted_class_index]
        predicted_probability = prediction[0][predicted_class_index]
        approx = round(predicted_probability * 100)


        vowels = ["Apple", "Avocado", "Orange"]
        if predicted_class_name in vowels:
            st.divider()
            st.write(f"Based on my prediction, your fruit is an {predicted_class_name} with a probability of {predicted_probability}")
            st.success(f"I am {approx}% sure that it's an {predicted_class_name}!🤩")


        else:
            st.divider()
            st.write(f"Based on my prediction, your fruit is a {predicted_class_name} with a probability of {predicted_probability}")
            st.success(f"I am {approx}% sure that it's a {predicted_class_name}!🤩")



    elif picture is None and fruit_image is not None:
        prediction_uploaded = model.predict(fruit_image)
        st.write("The probability that your fruit is any of the 10 fruits is shown below")

        fruits = []
        value = []
        for i in range(len(fruit_classes)):
            fruits.append(fruit_classes[i])
            value.append(float(prediction_uploaded[0][i]))

            if fruits == fruit_classes:
                me = pd.DataFrame({"FRUIT": fruits, 
                                "MY PROBABILITY PREDICTION": value})
                
                st.dataframe(me, hide_index=True)


        predicted_class_index_uploaded = np.argmax(prediction_uploaded)
        predicted_class_name_uploaded = fruit_classes[predicted_class_index_uploaded]
        predicted_probability_uploaded = prediction_uploaded[0][predicted_class_index_uploaded]
        approx_uploaded = round(predicted_probability_uploaded * 100)


        names = ["Apple", "Avocado", "Orange"]
        if predicted_class_name_uploaded in names:
            st.divider()
            st.write(f"Based on my prediction, your fruit is an {predicted_class_name_uploaded} with a probability of {predicted_probability_uploaded}")
            st.success(f"I am {approx_uploaded}% sure that it's an {predicted_class_name_uploaded}!🤩")


        else:
            st.divider()
            st.write(f"Based on my prediction, your fruit is a {predicted_class_name_uploaded} with a probability of {predicted_probability_uploaded}")
            st.success(f"I am {approx_uploaded}% sure that it's a {predicted_class_name_uploaded}!🤩")


    else:
        st.warning("Ooops! You have to upload just one image!")

