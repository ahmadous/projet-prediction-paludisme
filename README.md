# 🩸 Détection du Paludisme par Deep Learning

Système de détection automatique du paludisme à partir d'images de **cellules
sanguines**. Un modèle de réseau de neurones convolutif (CNN) analyse une image
de cellule et indique si elle est **infectée** ou **saine**, avec un pourcentage
de confiance.

Le projet est composé de deux parties :

- **Backend** — une API Flask qui charge le modèle Keras et expose l'endpoint de
  prédiction.
- **Frontend** — une interface web Vue 3 pour envoyer une image et visualiser le
  diagnostic.

---

## 🏗️ Architecture

```
┌─────────────────────┐        POST /predict (image)        ┌──────────────────────┐
│   Frontend (Vue 3)  │  ────────────────────────────────►  │   Backend (Flask)    │
│   Vite · port 5173  │                                     │   port 5000          │
│                     │  ◄────────────────────────────────  │   + modèle CNN Keras │
│  Upload · Aperçu    │     { classe, prediction, ... }     │                      │
└─────────────────────┘                                     └──────────────────────┘
```

```
projet-prediction-paludisme/
├── backend/                 # API Flask + modèle
│   ├── app.py               # Endpoints /, /health, /predict
│   ├── model.pkl            # Modèle CNN Keras (identique à model.joblib)
│   ├── model.joblib
│   ├── requirements.txt
│   └── README.md
├── frontend/                # Interface Vue 3 + Vite
│   ├── src/App.vue          # Composant principal
│   ├── package.json
│   └── README.md
├── image_palu/              # Images d'exemple (cellules saines / infectées)
├── palu_detection.ipynb     # Notebook d'entraînement du modèle
└── requirement.txt          # Dépendances Python (miroir de backend/requirements.txt)
```

---

## 🚀 Démarrage rapide

### 1. Backend (API Flask)

> **Python 3.11 requis** (Anaconda recommandé). TensorFlow ne fournit pas encore
> de wheel pour Python 3.13/3.14.

```bash
cd backend
pip install -r requirements.txt
python app.py
```

L'API démarre sur **http://localhost:5000**. Vérification :

```bash
curl http://localhost:5000/health     # {"status":"ok","model_loaded":true}
```

### 2. Frontend (interface web)

```bash
cd frontend
npm install
npm run dev
```

L'interface est disponible sur **http://localhost:5173**. Ouvre-la dans ton
navigateur, glisse une image de cellule, puis clique sur **« Analyser la
cellule »**.

---

## 🔌 API

### `POST /predict`

Analyse une image de cellule.

| Élément   | Détail                                        |
| --------- | --------------------------------------------- |
| Méthode   | `POST`                                        |
| Corps     | `multipart/form-data`, champ **`file`** = image |
| Formats   | png, jpg, jpeg, bmp, gif, tif                 |

**Réponse :**

```json
{
  "prediction": 0.9699,
  "prediction_percent": "96.99%",
  "classe": "Infecté"
}
```

**Exemple :**

```bash
curl -X POST -F "file=@image_palu/Parasitised.png" http://localhost:5000/predict
```

### Autres endpoints

| Endpoint      | Description                       |
| ------------- | -------------------------------- |
| `GET /`       | Informations et état du service  |
| `GET /health` | Sonde de disponibilité           |

Détails complets dans [backend/README.md](backend/README.md).

---

## 🧠 Modèle

CNN entraîné dans [`palu_detection.ipynb`](palu_detection.ipynb) :

- **Entrée** : image RGB redimensionnée en **64 × 64**, normalisée (`/255`)
- **Sortie** : `Dense(1, sigmoid)` → probabilité d'infection
  - `≥ 0.5` → **Infecté** (1)
  - `< 0.5` → **Sain** (0)
- Frameworks : **TensorFlow 2.16 / Keras 3.6**

---

## 🛠️ Stack technique

| Couche    | Technologies                                   |
| --------- | ---------------------------------------------- |
| Frontend  | Vue 3, Vite                                    |
| Backend   | Flask, flask-cors                              |
| ML        | TensorFlow / Keras, NumPy, Pillow              |

---

## ⚠️ Avertissement

Cet outil est un projet éducatif. Les résultats sont fournis à titre indicatif
et **ne remplacent pas un diagnostic médical** réalisé par un professionnel de
santé.
