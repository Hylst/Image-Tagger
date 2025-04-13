import io
import json
import logging
import re
import time
import os
import pathlib
import shutil
import unicodedata
from typing import Dict

from PIL import Image
from PIL.PngImagePlugin import PngInfo
from google.cloud import vision_v1
from iptcinfo3 import IPTCInfo
import pyexiv2

class ImageProcessor:
    def __init__(self, vision_client, gemini_model):
        self.vision_client = vision_client
        self.gemini_model = gemini_model

    @staticmethod
    def _sanitize_filename(title: str) -> str:
        """Normalise les caractères et supprime les accents"""
        normalized = unicodedata.normalize('NFKD', title)
        cleaned = normalized.encode('ASCII', 'ignore').decode('ASCII')
        return ''.join(c if c.isalnum() or c in ' -_' else '_' for c in cleaned.strip())[:50]

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
            original_path = pathlib.Path(image_path).resolve()
            image_bytes = self.resize_image(str(original_path))
            
            # Analyse des API
            vision_data = self._analyze_with_vision(image_bytes)
            gemini_data = self._analyze_with_gemini(image_bytes, vision_data)
            
            # Renommage et métadonnées
            new_path = self._rename_file(str(original_path), gemini_data.get('title', ''))
            metadata_status = self._write_metadata(new_path, gemini_data)
            
            return {
                "original_file": original_path.name,
                "new_file": pathlib.Path(new_path).name,
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
        """Renommage sécurisé pour Windows avec gestion des accents"""
        try:
            src = pathlib.Path(original_path)
            sanitized = ImageProcessor._sanitize_filename(title)
            ext = src.suffix.lower()
            dst = src.with_name(f"{sanitized}{ext}")

            # Gestion des doublons
            counter = 1
            while dst.exists():
                dst = src.with_name(f"{sanitized}_{counter}{ext}")
                counter += 1

            shutil.move(str(src), str(dst))
            return str(dst.resolve())
        except Exception as e:
            logging.error(f"Échec renommage : {str(e)}")
            return original_path

@staticmethod
def _write_metadata(image_path: str, metadata: dict) -> bool:
    """Écriture robuste des métadonnées pour tous formats"""
    try:
        path = pathlib.Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {path}")

        # Conversion spéciale pour pyexiv2 sous Windows
        win_path = bytes(path).decode('utf-8', 'surrogateescape')
        success = True

        try:
            # Écriture XMP/ITPC avec pyexiv2 (principal)
            with pyexiv2.Image(win_path) as img:
                # XMP Standard
                img.modify_xmp({
                    'Xmp.dc.title': metadata.get('title', ''),
                    'Xmp.photoshop.Headline': metadata.get('title', ''),
                    'Xmp.dc.description': metadata.get('description', ''),
                    'Xmp.dc.subject': metadata.get('keywords', []),
                    'Xmp.photoshop.Category': metadata.get('main_genre', ''),
                    'Xmp.photoshop.SupplementalCategories': [metadata.get('secondary_genre', '')]
                })

                # IPTC pour JPG
                if path.suffix.lower() in ['.jpg', '.jpeg']:
                    img.modify_iptc({
                        'Iptc.Application2.ObjectName': metadata.get('title', ''),
                        'Iptc.Application2.Headline': metadata.get('title', ''),
                        'Iptc.Application2.Caption': metadata.get('description', ''),
                        'Iptc.Application2.Keywords': metadata.get('keywords', [])
                    })

        except Exception as pyexiv_error:
            logging.warning(f"Erreur pyexiv2 : {str(pyexiv_error)}")
            success = False

        # Double écriture pour PNG avec Pillow (fallback)
        if path.suffix.lower() == '.png':
            try:
                with Image.open(path) as img:
                    # Conversion en RGB si nécessaire
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Écriture XMP dans PNG via PngInfo
                    png_info = PngInfo()
                    xmp_packet = f"""
                    <x:xmpmeta xmlns:x="adobe:ns:meta/">
                        <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
                            <rdf:Description rdf:about=""
                                xmlns:dc="http://purl.org/dc/elements/1.1/"
                                dc:title="{metadata.get('title', '')}"
                                dc:description="{metadata.get('description', '')}"
                                dc:subject="{','.join(metadata.get('keywords', []))}"/>
                        </rdf:RDF>
                    </x:xmpmeta>
                    """
                    png_info.add_text('XML:com.adobe.xmp', xmp_packet)
                    img.save(path, pnginfo=png_info, optimize=True)
                    success = True

            except Exception as pillow_error:
                logging.error(f"Erreur Pillow PNG : {str(pillow_error)}")
                success = False

        return success

    except Exception as e:
        logging.error(f"Échec global métadonnées [{path.name}] : {str(e)}")
        return False