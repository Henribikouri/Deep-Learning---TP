#***** auteur : HENRI BIKOURI*****

from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import keras
import os

APP_PORT = int(os.environ.get("PORT", 5000))
MODEL_PATH = os.environ.get("MODEL_PATH", "mnist_model.h5")

app = Flask(__name__)

# Chargement du modèle (au démarrage)
model = keras.models.load_model(MODEL_PATH)

# Helper: vérifier et transformer l'entrée en array (attendue shape: (1, 784) ou (28,28))
def preprocess_image(image):
    arr = np.array(image, dtype=np.float32)

    # Cas: image donnée en vecteur 784
    if arr.ndim == 1 and arr.size == 784:
        arr = arr.reshape((1, 28, 28))
    # Cas: image donnée en 2D (28,28)
    elif arr.ndim == 2 and arr.shape == (28, 28):
        arr = arr.reshape((1, 28, 28))
    # Cas: image fournie déjà normalisée entre 0-1 ou 0-255
    else:
        raise ValueError("Image must be shape (28,28) or flat length 784")

    # Normalisation
    if arr.max() > 1.0:
        arr = arr / 255.0

    arr = arr.reshape((1, 784)).astype(np.float32)
    return arr

@app.route("/predict", methods=["POST"])
def predict():
    if not request.is_json:
        return jsonify({"error": "Request content-type must be application/json"}), 400

    data = request.get_json()
    if "image" not in data:
        return jsonify({"error": "No image provided (JSON key 'image' missing)"}), 400

    try:
        image_data = preprocess_image(data["image"])
    except Exception as e:
        return jsonify({"error": f"Invalid image format: {str(e)}"}), 400

    # Prediction
    preds = model.predict(image_data)            # shape (1, 10)
    predicted_class = int(np.argmax(preds, axis=1)[0])
    probabilities = preds.tolist()[0]

    return jsonify({
        "prediction": predicted_class,
        "probabilities": probabilities
    }), 200

if __name__ == "__main__":
    # Mode développement (not for prod): Flask dev server
    app.run(host="0.0.0.0", port=APP_PORT, debug=False)
