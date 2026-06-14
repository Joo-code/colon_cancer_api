import gc
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
import io
import os

# FIXED: Changed single underscores to standard double underscores
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

        # 1. Clear session and run garbage collection before tracking new arrays
        tf.keras.backend.clear_session()
        gc.collect()

        # Decode base64 image
        image_bytes = base64.b64decode(data["image"])
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Preprocess
        processed_image = preprocess_image(image)

        # Predict
        prediction = model.predict(processed_image)
        
        # SAFE EXTRACTION: Extract the scalar value out of the array safely
        raw_prob = float(prediction[0][0])

        # Decision logic
        if raw_prob > 0.5:
            result = "adenocarcinoma"
            confidence = raw_prob
            recommendation = "High risk detected. Please consult a doctor immediately."
        else:
            result = "normal"
            confidence = 1.0 - raw_prob
            recommendation = "No cancer detected. Routine check recommended."

        # Compile JSON payload
        response_data = {
            "predictionResult": result,
            "predictionConfidence": round(confidence * 100, 2),
            "recommendation": recommendation,
            "visualization": {
                "probability_normal": round((1.0 - raw_prob) * 100, 2),
                "probability_cancer": round(raw_prob * 100, 2)
            }
        }

        # 2. Aggressively delete variable allocations to free up RAM instantly
        del image_bytes
        del image
        del processed_image
        del prediction
        
        # 3. Final memory clearance before response release
        tf.keras.backend.clear_session()
        gc.collect()

        return jsonify(response_data)

    except Exception as e:
        # Emergency memory cleanup during exception handling
        tf.keras.backend.clear_session()
        gc.collect()
        return jsonify({"error": str(e)}), 500


# ----------------------------
# RUN (RENDER COMPATIBLE)
# ----------------------------
# FIXED: Changed single underscores to double underscores
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)