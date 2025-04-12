import io
import json
import logging
import re
import time
import os
from typing import Dict
from PIL import Image
from google.cloud import vision_v1
from iptcinfo3 import IPTCInfo

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
            logging.warning(f"Échec du redimensionnement : {str(e)}")
            with open(image_path, "rb") as f:
                return f.read()

    def process_single_image(self, image_path: str) -> Dict:
        """Traite une image complète"""
        try:
            start_time = time.time()
            original_path = os.path.abspath(image_path)
            image_bytes = self.resize_image(image_path)
            
            # Analyse des API
            vision_data = self._analyze_with_vision(image_bytes)
            gemini_data = self._analyze_with_gemini(image_bytes, vision_data)
            
            # Renommage et métadonnées
            new_path = self._rename_file(original_path, gemini_data.get('title', ''))
            metadata_status = self._write_iptc_metadata(new_path, gemini_data.get('title', ''))
            
            return {
                "original_file": os.path.basename(original_path),
                "new_file": os.path.basename(new_path),
                "path": new_path,
                **gemini_data,
                "metadata_written": metadata_status,
                "processing_time": time.time() - start_time
            }
            
        except Exception as e:
            logging.error(f"Échec du traitement : {str(e)}")
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
            logging.error(f"Erreur Gemini : {str(e)}")
            return {}

    @staticmethod
    def _parse_gemini_response(text: str) -> Dict:
        """Extrait le JSON de la réponse Gemini"""
        try:
            json_str = re.search(r'({.*})', text, re.DOTALL).group(1)
            return json.loads(json_str)
        except Exception as e:
            logging.error(f"Échec de l'analyse JSON : {str(e)}")
            return {}

    @staticmethod
    def _rename_file(original_path: str, title: str) -> str:
        """Renomme le fichier avec le titre généré"""
        try:
            # Nettoyage du titre
            sanitized = "".join(c if c.isalnum() or c in " -_." else "_" for c in title.strip())
            sanitized = sanitized[:50].strip()  # Limite à 50 caractères
            
            ext = os.path.splitext(original_path)[1].lower()
            new_name = f"{sanitized}{ext}"
            new_path = os.path.join(os.path.dirname(original_path), new_name)
            
            # Gestion des doublons
            counter = 1
            while os.path.exists(new_path):
                new_name = f"{sanitized}_{counter}{ext}"
                new_path = os.path.join(os.path.dirname(original_path), new_name)
                counter += 1
                
            os.rename(original_path, new_path)
            return new_path
        except Exception as e:
            logging.error(f"Échec du renommage : {str(e)}")
            return original_path

    @staticmethod
    def _write_iptc_metadata(self, image_path: str, title: str) -> bool:
        """Écrit le titre dans les métadonnées IPTC"""
        try:
            if image_path.lower().endswith(('.jpg', '.jpeg')):
                # Gestion JPG avec iptcinfo3
                info = IPTCInfo(image_path, force=True)
                info['object name'] = title
                info['caption/abstract'] = title
                info.save()
                return True
                
            elif image_path.lower().endswith('.png'):
                # Gestion PNG avec Pillow
                with Image.open(image_path) as img:
                    img.info['iptc_data'] = title.encode('utf-8')
                    img.save(image_path)
                return True
                
            return False
            
        except Exception as e:
            logger.warning(f"Échec métadonnées IPTC ({os.path.basename(image_path)}): {str(e)}")
            return False