import os

from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    GIGACHAT_CREDENTIALS: str = os.getenv("GIGACHAT_CREDENTIALS", "sample")
    MODEL_SERVER: str = "ollama"  # Default to Ollama
    MODEL_NAME: str = "gemma:2b-instruct-fp16" # Default to gemma:2b-instruct-fp16
    OLLAMA_API_BASE: str = os.getenv("OLLAMA_API_BASE", "http://192.168.1.198:11434")
    EMBEDDING_MODEL_NAME: str = "intfloat/multilingual-e5-small"
    DATA_FILEPATH: str = "I:\\itmo\\large-language-sommelier\\data\\wine_desc.csv" # default for wine
    DATA_SOURCE_COLUMN: str = "Название" # default for wine
    PROMPT_TEMPLATE: str = (
        "Ты являешься профессиональным сомелье и помощником по подбору вин. "
        "Используй приведенные ниже фрагменты из извлеченного контекста, чтобы ответить на вопрос и помочь пользователю. "
        "Если ты не знаешь ответа, просто скажи, что ты не знаешь. "
        "Предлагай максимум три варианта.\n"
        "Вопрос: {question}\n"
        "Контекст: {context}\n"
        "Ответ:"
    )
    RAG_DATA_DIR: str = "data/"
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()