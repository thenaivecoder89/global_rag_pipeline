from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

response = client.embeddings.create(
    model="text-embedding-3-small",
    input="test embedding dimension"
)

embedding = response.data[0].embedding

print(f"Embedding dimension: {len(embedding)}")