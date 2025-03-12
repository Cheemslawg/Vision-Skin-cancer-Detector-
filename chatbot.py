# chatbot.py
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

class ChatAssistant:
    def __init__(self):
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        self.model = genai.GenerativeModel('gemini-pro')
        
        self.context = """
        You are a dermatology expert assistant specializing in skin cancer. Follow these rules:
        - Provide clear, medical information about skin cancer types (melanoma, BCC, SCC)
        - Explain differences between benign and malignant lesions
        - Offer prevention tips and early detection methods
        - Always recommend consulting a qualified dermatologist
        - Avoid making diagnoses
        - Keep responses under 500 characters
        """

    def get_response(self, user_input):
        try:
            prompt = f"{self.context}\n\nUser: {user_input}\nAssistant:"
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error: {str(e)}"