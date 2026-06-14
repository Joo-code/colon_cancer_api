from flask import Flask, request, jsonify
from flask_cors import CORS 
import tensorflow as tf 
import numpy as np 
from PIL import Image
import base64
import io 

app = Flask(__name__)
CORS(app)

model = None

print("Loading model...")
model = tf.keras.models.load_model("colon_cancer_model_fixed.keras", compile=False)
print("Model loaded successfully!")

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224,224))
    img_array = np.array(image).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route("/")
def home():
    return "Colon Cancer API is running"

@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()

        if not data or "image" not in data:
            return jsonify({"error": "No image uploaded"}), 400

        image_bytes = base64.b64decode(data["image"])
        image = Image.open(io.BytesIO(image_bytes))

        img_array = preprocess_image(image)

        prediction = model.predict(img_array, verbose=0)
        cnn_prob = float(prediction[0][0])

        if cnn_prob > 0.5:
            diagnosis = "adenocarcinoma"
            confidence = cnn_prob
            recommendation = "High risk detected. Immediate medical consultation recommended."
        else:
            diagnosis = "normal"
            confidence = 1- cnn_prob
            recommendation = "No signs of cancer detected. Routine monitoring advised."

        visualization = {
            "probability_normal": round((1-cnn_prob) * 100, 2),
            "probability_cancer": round(cnn_prob * 100, 2)
        }
    
        return jsonify({
            "predictionResult": diagnosis,
            "predictionConfidence": round(confidence * 100, 2),
            "recommendation": recommendation,
            "visualization": visualization
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)