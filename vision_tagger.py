import os
import json
from google.cloud import vision_v1
from google.oauth2 import service_account
from typing import List, Dict

class ImageTagger:
    def __init__(self, credentials_path: str):
        self.credentials = service_account.Credentials.from_service_account_file(credentials_path)
        self.client = vision_v1.ImageAnnotatorClient(credentials=self.credentials)

    def analyze_image(self, image_path: str) -> Dict:
        with open(image_path, "rb") as f:
            content = f.read()

        image = vision_v1.Image(content=content)
        features = [
            {"type_": vision_v1.Feature.Type.LABEL_DETECTION},
            {"type_": vision_v1.Feature.Type.WEB_DETECTION},
        ]

        response = self.client.annotate_image({
            "image": image,
            "features": features
        })

        return self._format_response(response, image_path)

    def _format_response(self, response, image_path: str) -> Dict:
        # Extraction des labels principaux
        labels = [label.description for label in response.label_annotations[:10]]
        
        # Détection de contexte web
        web_entities = [entity.description for entity in response.web_detection.web_entities[:5]]
        
        # Génération des métadonnées structurées
        return {
            "file_name": os.path.basename(image_path),
            "file_path": os.path.abspath(image_path),
            "title": " ".join(labels[:3]),  # 3 premiers labels comme titre
            "description": " ".join(labels + web_entities)[:200],  # Combinaison labels + entités web
            "genre": web_entities[0] if web_entities else "General",
            "keywords": ", ".join(labels + web_entities)
        }

def process_folder(folder_path: str, credentials_path: str, output_file: str = "results.json"):
    tagger = ImageTagger(credentials_path)
    results = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg')):
            full_path = os.path.join(folder_path, filename)
            try:
                result = tagger.analyze_image(full_path)
                results.append(result)
                print(f"Processed: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Path to image folder")
    parser.add_argument("--credentials", default="credentials.json", help="Path to GCP credentials")
    args = parser.parse_args()

    process_folder(args.folder, args.credentials)