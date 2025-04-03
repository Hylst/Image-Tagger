import io
import json
import logging
import re
import time
import os
from PIL import Image
from typing import Dict
from google.cloud import vision_v1

class ImageProcessor:
    def __init__(self, vision_client, gemini_model):
        self.vision_client = vision_client
        self.gemini_model = gemini_model

    @staticmethod
    def resize_image(image_path: str, max_size: int = 1024) -> bytes:
        """Redimensionne et convertit les images problématiques"""
        try:
            with Image.open(image_path) as img:
                if img.mode in ['P', 'RGBA']:
                    img = img.convert('RGB')
                
                img.thumbnail((max_size, max_size))
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=85)
                return buffer.getvalue()
        except Exception as e:
            logging.warning(f"Resize failed: {str(e)}")
            with open(image_path, "rb") as f:
                return f.read()

    def process_single_image(self, image_path: str) -> Dict:
        """Traite une image complète"""
        try:
            start_time = time.time()
            image_bytes = self.resize_image(image_path)
            
            # Analyse Vision API
            vision_data = self._analyze_with_vision(image_bytes)
            
            # Analyse Gemini
            gemini_data = self._analyze_with_gemini(image_bytes, vision_data)
            
            return {
                "file": os.path.basename(image_path),
                "path": os.path.abspath(image_path),
                **gemini_data,
                "processing_time": time.time() - start_time
            }
            
        except Exception as e:
            logging.error(f"Processing failed: {str(e)}")
            return {"error": str(e)}

    def _analyze_with_vision(self, image_bytes: bytes) -> Dict:
        """Appel Vision API"""
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

    def _analyze_with_gemini(self, image_bytes: bytes, vision_data: Dict) -> Dict:
        """Appel Gemini API avec prompt structuré"""
        prompt = """Analyse cette image et retourne un JSON avec :
        {
            "title": "Titre (3-7 mots)",
            "description": "Description détaillée",
            "main_genre": "Genre principal",
            "secondary_genre": "Sous-genre",
            "keywords": ["mot1", "mot2"]
        }"""
        
        try:
            response = self.gemini_model.generate_content([
                prompt,
                {"mime_type": "image/jpeg", "data": image_bytes}
            ])
            
            return self._parse_gemini_response(response.text)
        except Exception as e:
            logging.error(f"Gemini error: {str(e)}")
            return {}

    @staticmethod
    def _parse_gemini_response(text: str) -> Dict:
        """Extrait le JSON de la réponse Gemini"""
        try:
            json_str = re.search(r'({.*})', text, re.DOTALL).group(1)
            return json.loads(json_str)
        except Exception as e:
            logging.error(f"JSON parsing failed: {str(e)}")
            return {}