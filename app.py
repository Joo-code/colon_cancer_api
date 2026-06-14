from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
import io
import os

app = Flask(__name__)
CORS(app)

# ----------------------------
# LOAD MODEL (RENDER SAFE)
# ----------------------------
MODEL_PATH = "colon_cancer_model_clean.keras"

print("Loading model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully!")

# ----------------------------
# IMAGE PREPROCESSING
# ----------------------------
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ----------------------------
# ROUTES
# ----------------------------
@app.route("/")
def home():
    return "Colon Cancer API is running 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data or "image" not in data:
            return jsonify({"error": "No image provided"}), 400

        # Decode base64 image
        image_bytes = base64.b64decode(data["image"])
        image = Image.open(io.BytesIO(image_bytes))

        # Preprocess
        img_array = preprocess_image(image)

        # Predict
        prediction = model.predict(img_array, verbose=0)[0][0]

        # Decision logic
        if prediction > 0.5:
            result = "adenocarcinoma"
            confidence = float(prediction)
            recommendation = "High risk detected. Please consult a doctor immediately."
        else:
            result = "normal"
            confidence = float(1 - prediction)
            recommendation = "No cancer detected. Routine check recommended."

        return jsonify({
            "predictionResult": result,
            "predictionConfidence": round(confidence * 100, 2),
            "recommendation": recommendation,
            "visualization": {
                "probability_normal": round((1 - prediction) * 100, 2),
                "probability_cancer": round(prediction * 100, 2)
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ----------------------------
# RUN (RENDER COMPATIBLE)
# ----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)