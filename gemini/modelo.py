import google.generativeai as genai
from dotenv import load_dotenv
import os

# Carrega a variável de ambiente do arquivo .env (se você estiver usando .env)
load_dotenv()

# Configura a API key
chave_api = os.getenv("GEMINI_API_KEY")

genai.configure(api_key = chave_api)

# Inicia o modelo
model = genai.GenerativeModel("gemini-2.0-flash")

# rotina para enviar pergunta ao modelo
def responder_pergunta(pergunta: str) -> str:
    resposta = model.generate_content(pergunta)
    return resposta.text.strip()
