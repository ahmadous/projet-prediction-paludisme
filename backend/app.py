from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
from PIL import Image

app = Flask(__name__)
CORS(app)

# Charger le modèle depuis le fichier model.pkl
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    # Vérifier qu'un fichier a bien été envoyé
    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier envoyé'}), 400

    file = request.files['file']
    try:
        # Charger et prétraiter l'image
        img = Image.open(file.stream).convert('RGB')
        # Adapter la taille à celle attendue par votre modèle
        img = img.resize((64, 64))
        
        # Convertir en array numpy et normaliser
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Ajouter la dimension batch => shape (1, 64, 64, 3)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Faire la prédiction
        prediction = model.predict(img_array)  # ex: [[0.0026472194]]
        
        # Extraire la valeur float (probabilité d'être infecté si Dense(1, sigmoid))
        prediction_value = float(prediction[0][0])
        pourcentage = prediction_value * 100
        
        # Déterminer la classe (avec un seuil de 0.5)
        classe = "Infecté" if prediction_value >= 0.5 else "Sain"

        # Renvoyer les résultats
        return jsonify({
            'prediction': prediction_value,             # ex: 0.0026472194
            'prediction_percent': f"{pourcentage:.2f}%",# ex: "0.26%"
            'classe': classe                            # "Sain" ou "Infecté"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Lancement du serveur Flask
    app.run(debug=True)
