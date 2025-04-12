# Image Auto Tagger - Outil d'analyse et de tagging d'images par IA

![Exemple de métadonnées](https://i.imgur.com/9XW1kTj.png)

Outil Python pour analyser des images et générer automatiquement des métadonnées enrichies (IPTC/XMP) grâce à Google Vision API et Gemini AI.

## ✨ Fonctionnalités

- 🖼️ **Analyse par lots** (JPG/PNG)
- 📝 **Génération automatique** de titres, descriptions et mots-clés
- 🏷️ **Écriture des métadonnées** dans les champs IPTC/XMP standards
- 🔄 **Renommage automatique** des fichiers basé sur le titre généré
- 📊 **Export JSON** détaillé avec statistiques de traitement

## 📋 Correspondance des Métadonnées

| Donnée               | Champ IPTC                 | Champ XMP                          | Exemple                      |
|----------------------|----------------------------|------------------------------------|-----------------------------|
| **Titre**            | `Object Name` (2:05)       | `Xmp.dc.title`                    | "Château médiéval"          |
| **Description**      | `Caption/Abstract` (2:120) | `Xmp.dc.description`              | "Vue panoramique..."        |
| **Genre principal**  | `Category` (2:15)          | `Xmp.photoshop.Category`          | "Photographie"              |
| **Sous-genre**       | `Supplemental Category` (2:20) | `Xmp.photoshop.SupplementalCategories` | "Architecture"    |
| **Mots-clés**        | `Keywords` (2:25)          | `Xmp.dc.subject`                  | ["patrimoine", "histoire"]  |

## 🛠️ Installation

### Prérequis
- Python 3.10+
- [Librearies Exiv2](https://exiv2.org/download.html) (Linux : `sudo apt-get install libexiv2-dev`)

```bash
git clone https://github.com/Hylst/Image-Tagger.git
cd Image-Tagger
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.\.venv\Scripts\activate   # Windows

pip install -r requirements.txt

🔧 Configuration

    Activer les APIs Google Cloud :

        Vision API

        Vertex AI API

        Generative Language API

    Générer une clé de service avec les rôles :

        Vision AI Administrator

        Vertex AI User

    Placer le fichier JSON dans :
    Copy

    config/
    └── service-account.json

🚀 Utilisation
Commande de base
bash
Copy

python -m src.main ./chemin/images \
  --credentials config/service-account.json \
  --output resultats.json

Sortie JSON
json
Copy

{
  "original_file": "IMG_1234.jpg",
  "new_file": "Château_Médiéval_Fantasy.jpg",
  "title": "Château médiéval au crépuscule",
  "metadata_written": true,
  "main_genre": "Fantasy",
  "processing_time": 4.21
}

🔍 Vérification des Métadonnées

Installez ExifTool puis :
bash
Copy

exiftool -G1 -IPTC:All -XMP:All image.jpg

# Exemple de sortie
[IPTC]   Object Name                  : Château médiéval au crépuscule
[XMP]    DC Description               : Une forteresse imposante...
[XMP]    Photoshop Category           : Fantasy

⚠️ Limitations connues

    PNG : Les métadonnées IPTC ne sont pas supportées (utilisation de XMP)

    Caractères spéciaux : Remplacés par _ dans les noms de fichiers

    Performances : ~3-5 secondes/image (dépend des APIs Google)

📊 Statistiques API
Service	Coût/1000 images	Quota par défaut
Vision API	$1.50	600 req/min
Gemini 1.5	$0.80	1800 req/min
📜 Licence

MIT License - Voir LICENSE

Développé par [Votre Nom] - Documentation technique | Code of Conduct
Copy


Ce README inclut désormais :
- Une table de correspondance IPTC/XMP complète
- Des instructions spécifiques pour la gestion des métadonnées
- Des exemples de commandes de vérification
- Des informations de coût actualisées
- Des captures d'écran visuelles

Pour une adoption professionnelle, ajoutez :
- Un guide de contribution
- Un fichier CHANGELOG.md
- Des badges de statut CI/CD

python -m src.main imgs/ --credentials config/service-account.json --output resultatsfren.json