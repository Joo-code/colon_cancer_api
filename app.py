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
# CLINICAL RISK
# ----------------------------
def calculate_clinical_risk(
    age,
    height,
    weight,
    family_history,
    symptoms
):

    risk = 0.0

    if age > 60:
        risk += 0.10

    if family_history == 1:
        risk += 0.15

    try:
        bmi = weight /((height/100) **2)

        if bmi >= 30:
            risk += 0.10

    except:
        bmi = 0

    symptoms = symptoms.lower()

    if "blood" in symptoms:
        risk += 0.25

    if "weight loss" in symptoms:
        risk += 0.15

    if "abdominal pain" in symptoms:
        risk += 0.10

    return min(risk, 1.0)

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

        if not data:
            return jsonify({
                "error": "No data provided"
            }), 400
        
        if "image" not in data:
            return jsonify({
                "error": "No image provided"
            }), 400

        # Bersihkan memori sebelum bermula
        gc.collect()

        age = int(data.get("age", 0))

        height = float(data.get("height", 0))

        weight = float(data.get("weight", 0))

        family_history = int(
            data.get("family_history", 0)
        )

        symptoms = str(
            data.get("symptoms", "")
        )

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

        normal_probability = raw_prob

        cancer_probability = ( 1.0 - raw_prob)

        cnn_prob = cancer_probability

        clinical_risk = (
            calculate_clinical_risk(
                age,
                height,
                weight,
                family_history,
                symptoms
            )
        )

        final_score = (
            (0.8 * cnn_prob) + (0.2 * clinical_risk)
        )

        if final_score >= 0.5:
            result = "adenocarcinoma"
            confidence = final_score
            recommendation = "High risk detected. Please consult a doctor immediately."
        else:
            result = "normal"
            confidence = (1.0 - final_score)
            recommendation = "No cancer detected. Routine check recommended."

        print(f"CNN Normal: {normal_probability: .4f}")
        print(f"CNN Cancer: {cancer_probability:.4f}")
        print(f"Clinical Risk: {clinical_risk:.4f}")
        print(f"Final Score: {final_score:.4f}")
        print(f"Prediction: {result}")


        # Compile JSON payload
        response_data = {
            "predictionResult": result,
            "predictionConfidence": round(confidence * 100, 2),
            "recommendation": recommendation,
            "visualization": {
                        "probability_normal": round((1-final_score) * 100, 2),
                        "probability_cancer": round(final_score* 100, 2)
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