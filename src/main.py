import os
import sys
# Ajoute le chemin racine du projet
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import logging
import argparse
from src.config import initialize_apis
from src.image_processor import ImageProcessor



def main():
    parser = argparse.ArgumentParser(description="Image tagging with Google AI")
    parser.add_argument("input_path", help="Image file or directory")
    parser.add_argument("--credentials", required=True, help="GCP service account JSON")
    parser.add_argument("--output", default="results.json", help="Output file")
    parser.add_argument("--project", help="GCP project ID (détecté automatiquement si non spécifié)")
    
    args = parser.parse_args()

    # Récupération automatique du project_id
    if not args.project:
        with open(args.credentials) as f:
            credentials_data = json.load(f)
            args.project = credentials_data.get('project_id')
    
    if not args.project:
        raise ValueError("Project ID non trouvé dans les credentials et non spécifié")
    
    # Initialisation des APIs
    vision_client, gemini_model = initialize_apis(args.credentials, args.project)
    processor = ImageProcessor(vision_client, gemini_model)
    
    # Traitement
    results = []
    if os.path.isdir(args.input_path):
        for file in os.listdir(args.input_path):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                results.append(processor.process_single_image(os.path.join(args.input_path, file)))
    else:
        results.append(processor.process_single_image(args.input_path))
    
    # Sauvegarde
    with open(args.output, "w", encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logging.info(f"Processed {len(results)} images")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()