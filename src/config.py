import os
import google.generativeai as genai
from google.oauth2 import service_account
from google.cloud import vision_v1

def initialize_apis(credentials_path: str, project_id: str):
    """Initialise les APIs Google avec les identifiants"""
    credentials = service_account.Credentials.from_service_account_file(credentials_path)
    
    # Configuration Vision API
    vision_client = vision_v1.ImageAnnotatorClient(credentials=credentials)
    
    # Configuration Gemini
    # Configuration pour Gemini
    genai.configure(
        credentials=credentials
    )

    # Configuration pour Vision API
    vision_client = vision_v1.ImageAnnotatorClient(
        credentials=credentials,
    )

    return vision_client, genai.GenerativeModel('gemini-1.5-flash')
