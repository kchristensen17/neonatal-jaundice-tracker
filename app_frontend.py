import streamlit as st
import requests
from PIL import Image 
import io

# Flask Backend URL
FLASK_BACKEND_URL = "http://127.0.0.1:5000/predict"

# Streamlit UI
st.title("Neonatal Jaundice Tracker App")
st.write("Upload an Image to Get a Prediction.")

# File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image to bytes
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="JPEG")
    img_bytes=img_bytes.getvalue()

    # Send image to Flask backend 
    if st.button("Predict"):
        files = {"file": ("image.jpg", img_bytes, "image/jpeg")}
        response = requests.post(FLASK_BACKEND_URL, files=files)

        if response.status_code == 200:
            result = response.json()
            if "prediction" in result:
                st.success(f"Prediction: {result['prediction']}")
                st.info(f"Confidence: {result['Confidence']:.2%}")
            else:
                st.error("Prediction failed. Please try again")
        else:
            st.error(f"Error: {response.text}")

            