import gdown
from flask import Flask, request, jsonify 
import tensorflow as tf 
import numpy as np
from PIL import Image 
import requests 
from io import BytesIO
import os 

# Google Drive File ID for ML model 
GOOGLE_DRIVE_FILE_ID = "1KjBS2A2AbkcRxxbsaaR_kEmpBZIuauRV"

# Function to download the model from Google Drive
def download_model():
    url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
    model_path = "jaundice_detection_modelv2.keras"

    if not os.path.exists(model_path):
        gdown.download(url, model_path, quiet=False)

    return model_path

# Download and load the model
model_path = download_model()
model = tf.keras.models.load_model(model_path)

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return "Flask server is running"
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Read the image from the request 
        file = request.files["file"]
        img = Image.open(file).convert("RGB") # Convert to RGB
        img = img.resize((224,224)) # Resize to model's expected input 
        img_array = np.array(img) / 255.0 # Normalize pixel value 
        img_array = np.expand_dims(img_array, axis=0) # Add batch dimension

        # Run the model prediction
        predictions = model.predict(img_array)
        pred_value = float(predictions[0][0])

        # Convert the prediction to a human-readable result 
        if pred_value >=0.5:
            result = {"prediction": "Jaundice Detected", "Confidence": pred_value}
        else:
            result = {"prediction": "No Jaundice Detected", "Confidence": pred_value}
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})
if __name__ == "__main__":
    print("Starting Flask server....")
    app.run(debug=True)

     