import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

# Load the trained model
model = load_model('CNN_model.h5')

# Label Encoder for the rice varieties
label_encoder = {0: 'Arborio', 1: 'Basmati', 2: 'Ipsala', 3: 'Jasmine', 4: 'Karacadag'}

# Streamlit app
st.title("Rice Classification")
st.write("Upload images of rice to classify their variety.")

uploaded_files = st.file_uploader("Choose images...", type="jpg", accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Preprocess the image for prediction
        img = load_img(uploaded_file, target_size=(50, 50))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make the prediction
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence_score = np.max(prediction) * 100

        # Get the rice variety
        predicted_rice_variety = label_encoder[predicted_class]

        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=300*300)

        # Display the prediction
        st.write(f"**Rice Variety:** {predicted_rice_variety}")
        st.write(f"**Confidence Score:** {confidence_score:.2f}%")
        st.write("---")
