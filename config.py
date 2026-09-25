import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "huggingface")
HUGGINGFACE_MODEL = "all-MiniLM-L6-v2"

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "google")
DEFAULT_MODEL_NAME = "gemini-1.5-flash" if LLM_PROVIDER == "google" else "gpt-4o-mini"

NUM_TOPIC_CLUSTERS = 4
CHROMA_PERSIST_DIR = "./chroma_db"
