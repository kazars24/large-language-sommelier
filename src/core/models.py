from pydantic import BaseModel


class Query(BaseModel):
    question: str


class ModelChoice(BaseModel):
    model_server: str = "ollama"
    model_name: str = "gemma:2b-instruct-fp16"
