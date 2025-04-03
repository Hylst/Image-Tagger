import os
import json
import re
import time
import logging
from typing import Dict, List
from google.api_core import retry, exceptions
from google.cloud import vision_v1
import google.generativeai as genai
from PIL import Image
import io

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedImageTagger:
    def __init__(self, credentials_path: str, project_id: str):
        self.project_id = project_id
        self.credentials_path = credentials_path
        self._init_clients()

    @retry.Retry(
        predicate=retry.if_exception_type(
            exceptions.DeadlineExceeded,
            exceptions.ServiceUnavailable
        ),
        initial=1.0,
        maximum=10.0,
        multiplier=2.0,
        deadline=60.0
    )
    def _init_clients(self):
        """Initialise les clients GCP avec gestion de réessais"""
        try:
            # Configuration Vision API
            self.vision_client = vision_v1.ImageAnnotatorClient.from_service_account_file(
                self.credentials_path)
            
            # Configuration Vertex AI
            aiplatform.init(
                project=self.project_id,
                location="us-central1",
                credentials=self.credentials_path
            )
            self.vertex_client = aiplatform.gapic.PredictionServiceClient()
            
        except Exception as e:
            logger.error(f"Erreur d'initialisation : {str(e)}")
            raise

    def process_image(self, image_path: str) -> Dict:
        """Traite une image et retourne les métadonnées enrichies"""
        try:
            start_time = time.time()
            
            # Étape 1: Prétraitement de l'image
            resized_image = self._resize_image(image_path)
            
            # Étape 2: Analyse Vision API
            vision_data = self._vision_analysis(resized_image)
            
            # Étape 3: Analyse Gemini
            gemini_data = self._gemini_analysis(resized_image, vision_data)
            
            # Formatage des résultats
            result = self._format_result(image_path, vision_data, gemini_data)
            
            logger.info(f"Traitement réussi en {time.time()-start_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Échec du traitement : {str(e)}")
            return {
                "file": os.path.basename(image_path),
                "error": str(e)
            }

    def _resize_image(self, image_path: str, max_size: int = 1024) -> bytes:
        """Redimensionne l'image pour respecter les limites de l'API"""
        try:
            with Image.open(image_path) as img:
                img.thumbnail((max_size, max_size))
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=85)
                return buffer.getvalue()
        except Exception as e:
            logger.warning(f"Échec du redimensionnement : {str(e)}")
            with open(image_path, "rb") as f:
                return f.read()

    def _vision_analysis(self, image_bytes: bytes) -> Dict:
        """Analyse basique avec Vision API"""
        try:
            image = vision_v1.Image(content=image_bytes)
            response = self.vision_client.annotate_image({
                "image": image,
                "features": [
                    {"type_": vision_v1.Feature.Type.LABEL_DETECTION},
                    {"type_": vision_v1.Feature.Type.WEB_DETECTION}
                ]
            })

            return {
                "labels": [label.description for label in response.label_annotations],
                "web_entities": [entity.description for entity in response.web_detection.web_entities]
            }
        except Exception as e:
            logger.warning(f"Vision API a échoué : {str(e)}")
            return {"labels": [], "web_entities": []}

    @retry.Retry(
        deadline=30.0,
        initial=1.0,
        maximum=5.0,
        multiplier=2.0
    )
    def _gemini_analysis(self, image_bytes: bytes, vision_data: Dict) -> Dict:
        """Analyse avancée avec Gemini"""
        try:
            prompt = """Analyse cette image en détail et retourne un JSON VALIDE avec :
            {
                "title": "Titre créatif (3-7 mots)",
                "description": "Description détaillée en 2-3 phrases",
                "main_genre": "Genre principal (ex: Photographie, Art numérique...)",
                "secondary_genre": "Sous-genre (ex: Portrait fantastique...)",
                "keywords": ["liste", "de", "mots-clés"]
            }
            Conseils :
            - Sois précis sur les couleurs et la composition
            - Utilise des termes techniques si pertinent"""

            instance = {
                "structVal": {
                    "prompt": {"stringValue": prompt},
                    "image": {"bytesValue": image_bytes},
                    "temperature": {"floatValue": 0.4},
                    "max_output_tokens": {"intValue": 800}
                }
            }

            endpoint = f"projects/{self.project_id}/locations/us-central1/publishers/google/models/gemini-1.0-pro-vision"
            response = self.vertex_client.predict(
                endpoint=endpoint,
                instances=[instance],
                timeout=15.0
            )

            return self._parse_gemini_response(response.predictions[0])
            
        except Exception as e:
            logger.error(f"Échec de Gemini : {str(e)}")
            return {}

    def _parse_gemini_response(self, response) -> Dict:
        """Extrait et valide la réponse de Gemini"""
        try:
            raw_text = response["structVal"]["content"]["stringValue"]
            
            # Nettoyage de la réponse
            json_str = re.sub(r'[\s\S]*?({.*})[\s\S]*?', r'\1', raw_text, flags=re.DOTALL)
            json_str = json_str.replace("```json", "").replace("```", "").strip()
            
            result = json.loads(json_str)
            
            # Validation des champs
            required_fields = ["title", "description", "main_genre", "secondary_genre", "keywords"]
            for field in required_fields:
                if field not in result:
                    result[field] = "Non spécifié"
                    
            return result
            
        except json.JSONDecodeError:
            logger.error(f"Réponse JSON invalide : {raw_text[:200]}...")
            return {
                "title": "Non spécifié",
                "description": "Non spécifié",
                "main_genre": "Non spécifié",
                "secondary_genre": "Non spécifié",
                "keywords": []
            }

    def _format_result(self, image_path: str, vision_data: Dict, gemini_data: Dict) -> Dict:
        """Fusionne les données de Vision API et Gemini"""
        return {
            "file": os.path.basename(image_path),
            "path": os.path.abspath(image_path),
            "title": gemini_data.get("title", " ".join(vision_data["labels"][:3])),
            "description": gemini_data.get("description", " ".join(vision_data["web_entities"][:3])),
            "main_genre": gemini_data.get("main_genre", "Non spécifié"),
            "secondary_genre": gemini_data.get("secondary_genre", "Non spécifié"),
            "keywords": ", ".join(gemini_data.get("keywords", []) + vision_data["labels"][:3]),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

def main():
    """Point d'entrée principal"""
    import argparse
    parser = argparse.ArgumentParser(description="Analyse d'images avec Google Cloud Vision et Gemini")
    parser.add_argument("input", help="Fichier ou dossier d'images")
    parser.add_argument("--output", default="results.json", help="Fichier de sortie JSON")
    parser.add_argument("--credentials", required=True, help="Chemin des identifiants GCP")
    parser.add_argument("--project", required=True, help="ID du projet GCP")
    args = parser.parse_args()

    tagger = EnhancedImageTagger(args.credentials, args.project)
    results = []

    if os.path.isdir(args.input):
        for file in os.listdir(args.input):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                results.append(tagger.process_image(os.path.join(args.input, file)))
    else:
        results.append(tagger.process_image(args.input))

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Résultats sauvegardés dans {args.output}")

if __name__ == "__main__":
    main()