from dotenv import load_dotenv
import os
from google import genai

# Load variables from .env
load_dotenv()

# Read the API key
api_key = os.getenv("GEMINI_API_KEY")

# Create Gemini client
client = genai.Client(api_key=api_key)

# Ask Gemini a question
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Explain Retrieval-Augmented Generation in simple words."
)

print(response.text)