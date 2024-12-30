import os

# Langfuse
LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY", 'pk-lf-7d30e5df-41a6-4888-9602-2e5e19e6ef73')
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY", 'sk-lf-2195fe3e-eb64-4bdc-8891-072f603e528f')
LANGFUSE_HOST = os.environ.get("LANGFUSE_HOST", "http://localhost:3000")

# Ollama
OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE", "http://192.168.1.198:11434")

# HuggingFace Embeddings (For Ragas Evaluation)
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
