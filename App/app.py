import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

class_labels = {0: "Healthy", 1: "Leaf Blight", 2: "Yellow Mottle Virus"}
model_path = 'Model/blackPepper_mobileNet_model.h5'
model = load_model(model_path)

def page_home():
    st.title("Black Pepper Leaf Disease Detection")
    st.write("Explore our innovative black pepper leaf disease detection system, utilizing advanced technology to accurately identify and manage plant health for optimal yield and sustainability.")
    image_path = "C:/Git/black-pepper/images/blackPepper.jpg"  # Updated to a variable for clarity
    image = open(image_path, "rb").read()
    st.image(image, use_column_width=True)

    st.markdown("* * *")
    columns = st.columns((2, 1, 2))
    button_pressed = columns[1].button('Start Now')
    
    if button_pressed:
        st.session_state.page = "Detection"
        st.experimental_rerun() 

def detection(imagefile):
    st.markdown("* * *")
    
def page_Detection():
    st.title("Black Pepper Leaf Disease Detection")
    st.write("We understand the critical role black pepper plays in the global spice market and the devastating impact that diseases can have on its cultivation. Our innovative online platform is dedicated to empowering farmers, agronomists, and researchers with the ability to detect and diagnose diseases in black pepper plants with unprecedented accuracy and speed.")

    st.markdown("* * *")
    st.write("Upload a clear leaf image in JPEG/JOG/PNG format..")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    st.markdown("* * *")

    if uploaded_file is not None:
        image_display = Image.open(uploaded_file)
        st.image(image_display, use_column_width=True)

        img = preprocess_image(image_display)

        st.markdown("* * *")
        columns = st.columns((2, 1, 2))
        button_detection = columns[1].button('Click Here to Predict..')
        
        if button_detection:
            prediction = model.predict(img)
            predicted_class = np.argmax(prediction)
            predicted_label = class_labels[predicted_class]
            st.markdown("* * *")
            st.write("<div style='text-align:center'>Prediction Result: {}</div>".format(predicted_label), unsafe_allow_html=True)
            st.markdown("* * *")

def preprocess_image(img):
 
    img = img.resize((224, 224))
    img = np.array(img)
    if img.ndim == 2:
        img = np.stack((img,) * 3, axis=-1)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

def page_Treatment():
    st.title("Treatments for Diseases")
    st.write("treatments")

def sidebar_layout():
    st.sidebar.title("Menu")
    if st.sidebar.button("Home"):
        st.session_state.page = "Home"
    if st.sidebar.button("Detection"):
        st.session_state.page = "Detection"
    if st.sidebar.button("Treatment"):
        st.session_state.page = "Treatment"

def main():
    if 'page' not in st.session_state:
        st.session_state.page = "Home"

    sidebar_layout()
    
    if st.session_state.page == "Home":
        page_home()
    elif st.session_state.page == "Detection":
        page_Detection()
    elif st.session_state.page == "Treatment":
        page_Treatment()

if __name__ == "__main__":
    main()
