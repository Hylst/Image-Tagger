# Image Tagger - Documentation

## 📌 Fonctionnalités
- Analyse d'images par lot avec Google Vision API
- Génération de métadonnées enrichies via Gemini (Vertex AI)
- Export des résultats au format JSON
- Supporte JPG/PNG/WebP

## 🚀 Installation

### Prérequis
- Python 3.9+
- Compte Google Cloud Platform (GCP)
- Clé API JSON (voir section Configuration)

### 1. Installer les dépendances
```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows
pip install -r requirements.txt


🔑 Configuration des clés API
Étape 1 - Créer un projet GCP

    Allez sur Google Cloud Console
https://console.cloud.google.com/
    Créez un nouveau projet (ex: "Image-Tagger")

    Notez l'ID du projet

Étape 2 - Activer les APIs

    Activez ces services :

        Vision API

        Vertex AI API

        Generative Language API

Étape 3 - Créer un compte de service

    Allez dans IAM & Admin > Comptes de service

    Créez un compte avec :

        Nom : image-tagger-service

        Rôles :

            Vertex AI User

            Vision AI Administrator

Étape 4 - Générer la clé JSON

    Dans le compte de service créé :

        Onglet Clés > Ajouter une clé

        Format : JSON

    Téléchargez le fichier et placez-le dans :

    /config/
    └── gcp-credentials.json

🖥️ Utilisation
Lancer le traitement


python vision_gemini_tagger.py ./images \
  --credentials config/gcp-credentials.json \
  --project VOTRE_PROJECT_ID \
  --output results.json

python -m src.main imgs/ --credentials config/service-account.json --output resultatsfr.json

Structure de sortie (JSON)

{
  "file": "image.jpg",
  "path": "/chemin/absolu/image.jpg",
  "title": "Titre généré",
  "description": "Description détaillée...",
  "main_genre": "Genre principal",
  "secondary_genre": "Sous-genre",
  "keywords": "liste, de, mots-clés",
  "timestamp": "2024-04-05 12:00:00"
}

⚠️ Notes importantes

    Coûts GCP : Environ $0.50 pour 1000 images

    Taille max : 4MB par image

    Formats supportés : JPG, PNG, WebP

📚 Documentation technique

    Vision API Reference
https://cloud.google.com/vision/docs
    Vertex AI Pricing
https://cloud.google.com/vertex-ai/pricing

📄 License

MIT License - Voir LICENSE

---

### Arborescence recommandée

.
├── .venv/
├── config/
│ └── gcp-credentials.json # Clé API
├── images/ # Dossier des images à analyser
├── results.json # Résultats générés
└── .gitignore # Ignore les fichiers sensibles


Ce setup permet une configuration sécurisée et reproductible. N'oubliez pas d'ajouter `config/` et `.venv/` à votre `.gitignore` ! 🔒
