from pydantic import BaseModel


class EvaluationRequest(BaseModel):
    question: str
    rag_response: str
    docs: list = []
    critic_model: str = "gemma:2b-instruct-fp16"

class ModelChoice(BaseModel):
    model_server: str = "ollama"
    model_name: str = "gemma:2b-instruct-fp16"


class SummarizeRequest(BaseModel):
    text: str
    model: str
    model_server: str
