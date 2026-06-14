import gc
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import base64
import io
import os
import tflite_runtime.interpreter as tflite

app = Flask(__name__)
CORS(app)

# ----------------------------
# LOAD TF LITE MODEL
# ----------------------------
MODEL_PATH = "colon_cancer_model.tflite"

print("Loading TF Lite model...")
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Dapatkan info input & output tensor
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("TF Lite Model loaded successfully!")

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

        # Bersihkan memori sebelum bermula
        gc.collect()

        # Decode base64 image
        image_bytes = base64.b64decode(data["image"])
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Preprocess
        processed_image = preprocess_image(image)

        # Predict menggunakan TF Lite Interpreter
        interpreter.set_tensor(input_details[0]['index'], processed_image)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        
        # SAFE EXTRACTION: Ekstrak nilai perpuluhan dengan selamat
        raw_prob = float(prediction[0][0])

        # Decision logic
        if raw_prob >= 0.5:
            result = "normal"
            confidence = raw_prob
            recommendation = "No cancer detected. Routine check recommended."
        else:
            result = "adenocarcinoma"
            confidence = 1.0 - raw_prob
            recommendation = "High risk detected. Please consult a doctor immediately."

        probability_normal = round(raw_prob * 100, 2)
        probability_cancer = round((1.0 - raw_prob) * 100, 2)

        print(f"Raw Probability: {raw_prob}")
        print(f"Prediction Result: {result}")
        print(f"Normal Probability: {probability_normal}%")
        print(f"Cancer Probability: {probability_cancer}%")


        # Compile JSON payload
        response_data = {
            "predictionResult": result,
            "predictionConfidence": round(confidence * 100, 2),
            "recommendation": recommendation,
            "visualization": {
                "probability_normal": probability_normal,
                "probability_cancer": probability_cancer
            }
        }

        # Aggressively delete variable allocations to free up RAM instantly
        del image_bytes
        del image
        del processed_image
        del prediction
        
        # Final memory clearance
        gc.collect()

        return jsonify(response_data)

    except Exception as e:
        # Emergency memory cleanup
        gc.collect()
        return jsonify({"error": str(e)}), 500


# ----------------------------
# RUN (RENDER COMPATIBLE)
# ----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)