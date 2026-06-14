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
# MODEL ARCHITECTURE (must match training)
# ----------------------------
print("Loading model architecture...")

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# ----------------------------
# LOAD WEIGHTS
# ----------------------------
WEIGHTS_PATH = "weights.weights.h5"

if not os.path.exists(WEIGHTS_PATH):
    raise FileNotFoundError(f"Missing weights file: {WEIGHTS_PATH}")

model.load_weights(WEIGHTS_PATH)

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
    return "Colon Cancer API is running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data or "image" not in data:
            return jsonify({"error": "No image uploaded"}), 400

        # Decode image
        image_bytes = base64.b64decode(data["image"])
        image = Image.open(io.BytesIO(image_bytes))

        # Preprocess
        img_array = preprocess_image(image)

        # Predict
        prediction = model.predict(img_array, verbose=0)
        cnn_prob = float(prediction[0][0])

        # Decision logic
        if cnn_prob > 0.5:
            diagnosis = "adenocarcinoma"
            confidence = cnn_prob
            recommendation = "High risk detected. Immediate medical consultation recommended."
        else:
            diagnosis = "normal"
            confidence = 1 - cnn_prob
            recommendation = "No signs of cancer detected. Routine monitoring advised."

        return jsonify({
            "predictionResult": diagnosis,
            "predictionConfidence": round(confidence * 100, 2),
            "recommendation": recommendation,
            "visualization": {
                "probability_normal": round((1 - cnn_prob) * 100, 2),
                "probability_cancer": round(cnn_prob * 100, 2)
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ----------------------------
# RUN SERVER (Render compatible)
# ----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)