import time
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.outputs import LLMResult
from datasets import Dataset
from typing import List
from ragas.llms.base import BaseRagasLLM
from langchain_ollama.llms import OllamaLLM
from config import EMBEDDING_MODEL_NAME


class OllamaRagasLLM(BaseRagasLLM):
    """Custom Ragas LLM implementation for Ollama."""

    def __init__(self, ollama_instance: OllamaLLM):
        self.ollama = ollama_instance

    def generate_text(self, prompt: str, **kwargs) -> str:
        response = self.ollama.invoke(prompt)
        return response

    async def agenerate_text(self, prompt: str, **kwargs) -> str:
        response = await self.ollama.ainvoke(prompt)
        return response

    async def generate(
        self,
        prompts: List[str],
        n: int = 1,
        temperature: float = 0,
        **kwargs
    ) -> LLMResult:
        generations = []
        for prompt in prompts:
            response = await self.ollama.ainvoke(prompt)
            prompt_generations = [{"text": response, "output": response}]
            generations.append(prompt_generations)

        return LLMResult(generations=generations)

class SummaryEvaluator:
    """Class for evaluating summaries using Ragas."""

    def __init__(self, ollama_llm: OllamaLLM):
        self.ragas_llm = OllamaRagasLLM(ollama_llm)
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        self.metrics = [faithfulness, answer_relevancy]

    def evaluate_summary(self, text: str, summary: str) -> dict:
        """Evaluates the generated summary using Ragas and returns the scores."""

        data = {
            "question": ["Summarize this text"],
            "context": [text],
            "answer": [summary],
            "retrieved_contexts": [[text]]
        }
        dataset = Dataset.from_dict(data)

        try:
            ragas_scores = evaluate(
                dataset=dataset,
                metrics=self.metrics,
                llm=self.ragas_llm,
                embeddings=self.embeddings
            )
        except Exception as e:
            print(f"Evaluation error: {str(e)}")
            ragas_scores = {
                "faithfulness": [0.0],
                "answer_relevancy": [0.0]
            }

        return ragas_scores
