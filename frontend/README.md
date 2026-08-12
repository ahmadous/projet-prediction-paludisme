# Frontend — Détection du Paludisme

Interface Vue 3 + Vite qui consomme l'API Flask (`backend/app.py`) pour détecter
le paludisme à partir d'une image de cellule sanguine.

## Fonctionnalités

- Upload d'image par glisser-déposer ou sélecteur de fichiers
- Aperçu de la cellule avant analyse
- Appel de l'endpoint `POST /predict` du backend
- Affichage du diagnostic (**Sain** / **Infecté**), du pourcentage de
  probabilité d'infection et d'un indice de confiance

## Démarrage

### 1. Backend (dans un terminal séparé)

```bash
cd ../backend
python app.py          # démarre Flask sur http://localhost:5000
```

> Le modèle (`model.pkl`) est un CNN Keras. En plus de `requirement.txt`
> (Flask, flask-cors, Pillow, numpy), le backend a besoin de **tensorflow/keras**
> pour charger le modèle.

### 2. Frontend

```bash
npm install
npm run dev            # http://localhost:5173
```

## Configuration de l'URL du backend

Par défaut le frontend appelle `http://localhost:5000`. Pour pointer vers une
autre adresse, créer un fichier `.env.local` :

```
VITE_API_URL=http://mon-serveur:5000
```

## Build de production

```bash
npm run build          # génère dist/
npm run serve          # sert dist/ en local (vite preview)
```

## Contrat de l'API `/predict`

- **Requête** : `POST` multipart/form-data, champ `file` = image
- **Réponse** :
  ```json
  {
    "prediction": 0.9699,
    "prediction_percent": "96.99%",
    "classe": "Infecté"
  }
  ```
