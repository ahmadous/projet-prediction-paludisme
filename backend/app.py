"""
API de détection du paludisme à partir d'images de cellules sanguines.

Le modèle est un CNN Keras entraîné sur des images 64x64 RGB (normalisées /255),
avec une sortie Dense(1, sigmoid) : proche de 1 => "Infecté", proche de 0 => "Sain".

Endpoints :
  GET  /          -> informations et état du service
  GET  /health    -> sonde de disponibilité (JSON {status, model_loaded})
  POST /predict   -> prédiction sur une image (multipart/form-data, champ "file")
"""

import io
import os

import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, UnidentifiedImageError

# --- Configuration -----------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(BASE_DIR, "model.pkl"))
IMG_SIZE = 64  # Taille d'entrée attendue par le modèle (cf. notebook)
THRESHOLD = float(os.environ.get("THRESHOLD", "0.5"))  # Seuil de décision
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "gif", "tif", "tiff"}
MAX_CONTENT_LENGTH = 8 * 1024 * 1024  # 8 Mo max par requête

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
CORS(app)  # Autorise les appels depuis le frontend (autre origine)


# --- Chargement du modèle ----------------------------------------------------

def load_model(path):
    """Charge le modèle. Essaie pickle puis joblib (les deux formats existent)."""
    try:
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as pickle_err:
        try:
            import joblib
            return joblib.load(path)
        except Exception as joblib_err:
            raise RuntimeError(
                f"Impossible de charger le modèle depuis {path}. "
                f"pickle: {pickle_err} | joblib: {joblib_err}"
            )


model = load_model(MODEL_PATH)


# --- Utilitaires -------------------------------------------------------------

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess(image_bytes):
    """Transforme des octets d'image en tenseur (1, 64, 64, 3) normalisé."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)  # shape (1, 64, 64, 3)


# --- Endpoints ---------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "API de détection du paludisme",
        "model_loaded": model is not None,
        "input_size": [IMG_SIZE, IMG_SIZE],
        "threshold": THRESHOLD,
        "endpoints": {
            "GET /health": "état du service",
            "POST /predict": "prédiction (multipart/form-data, champ 'file')",
        },
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


@app.route("/predict", methods=["POST"])
def predict():
    # 1. Vérifier la présence du fichier
    if "file" not in request.files:
        return jsonify({"error": "Aucun fichier envoyé (champ 'file' manquant)"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Nom de fichier vide"}), 400
    if not allowed_file(file.filename):
        return jsonify({
            "error": "Format non supporté. Formats acceptés : "
                     + ", ".join(sorted(ALLOWED_EXTENSIONS))
        }), 400

    # 2. Prétraiter l'image
    try:
        img_array = preprocess(file.read())
    except UnidentifiedImageError:
        return jsonify({"error": "Fichier image invalide ou corrompu"}), 400
    except Exception as e:
        return jsonify({"error": f"Erreur de traitement de l'image : {e}"}), 400

    # 3. Prédire
    try:
        prediction = model.predict(img_array, verbose=0)
        prediction_value = float(np.asarray(prediction).flatten()[0])
    except Exception as e:
        return jsonify({"error": f"Erreur lors de la prédiction : {e}"}), 500

    pourcentage = prediction_value * 100
    classe = "Infecté" if prediction_value >= THRESHOLD else "Sain"

    return jsonify({
        "prediction": prediction_value,
        "prediction_percent": f"{pourcentage:.2f}%",
        "classe": classe,
    })


@app.errorhandler(413)
def too_large(_e):
    return jsonify({"error": "Fichier trop volumineux (max 8 Mo)"}), 413


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(host=host, port=port, debug=debug)
