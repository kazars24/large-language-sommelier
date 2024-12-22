from langchain_gigachat.chat_models import GigaChat
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from src.core.config import settings


def load_llm():
    giga = GigaChat(credentials=settings.GIGACHAT_CREDENTIALS, model=settings.MODEL_NAME, timeout=30, verify_ssl_certs=False)
    giga.verbose = False
    return giga


def load_embedding_model():
    embeddings_model = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL_NAME,
        multi_process=False,
        encode_kwargs={"normalize_embeddings": True})
    return embeddings_model
