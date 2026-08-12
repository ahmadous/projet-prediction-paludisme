# Backend — API de détection du paludisme

API Flask qui charge un CNN Keras et prédit si une cellule sanguine est
**Infectée** ou **Saine** à partir d'une image.

## Prérequis

**Python 3.11** (Anaconda recommandé). TensorFlow ne fournit pas encore de wheel
pour Python 3.13/3.14.

```bash
pip install -r requirements.txt
```

## Démarrage

```bash
python app.py
```

Le serveur écoute sur `http://127.0.0.1:5000`. Variables d'environnement
optionnelles :

| Variable      | Défaut        | Rôle                                  |
| ------------- | ------------- | ------------------------------------- |
| `HOST`        | `127.0.0.1`   | Adresse d'écoute                      |
| `PORT`        | `5000`        | Port                                  |
| `MODEL_PATH`  | `model.pkl`   | Chemin du modèle                      |
| `THRESHOLD`   | `0.5`         | Seuil de décision Infecté/Sain        |
| `FLASK_DEBUG` | `1`           | Mode debug (`0` pour désactiver)      |

## Endpoints

### `GET /`
Informations et état du service.

### `GET /health`
Sonde de disponibilité.
```json
{ "status": "ok", "model_loaded": true }
```

### `POST /predict`
Prédiction sur une image.

- **Requête** : `multipart/form-data`, champ `file` = image (png, jpg, …)
- **Réponse** :
  ```json
  {
    "prediction": 0.9699,
    "prediction_percent": "96.99%",
    "classe": "Infecté"
  }
  ```

Exemple :
```bash
curl -X POST -F "file=@../image_palu/Parasitised.png" http://localhost:5000/predict
```

## Modèle

CNN Keras entraîné dans `../palu_detection.ipynb` : entrée **64×64 RGB**
normalisée (`/255`), sortie `Dense(1, sigmoid)` où **1 = Infecté**, **0 = Sain**.
Les fichiers `model.pkl` et `model.joblib` contiennent le même modèle.
