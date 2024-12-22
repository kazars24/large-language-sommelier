import os

# Langfuse
LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.environ.get("LANGFUSE_HOST", "http://localhost:3000")

# Ollama
OLLAMA_API_BASE = "http://192.168.1.198:11434"

# HuggingFace Embeddings
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
