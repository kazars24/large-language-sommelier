from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.outputs import LLMResult
from datasets import Dataset
from typing import List
from ragas.llms.base import BaseRagasLLM
from langchain_ollama.llms import OllamaLLM
from langchain_community.chat_models.gigachat import GigaChat
from validation.config import EMBEDDING_MODEL_NAME
from src.retriever.splitter import UniversalDocumentSplitter
from src.core.config import settings
import streamlit as st

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

class RagasEvaluator:
    """Class for evaluating summaries using Ragas."""

    def __init__(self, ollama_llm: OllamaLLM, model_server: str, model_name: str, embedding_model_name: str = EMBEDDING_MODEL_NAME, debug_mode: bool = False):
        self.ragas_llm = OllamaRagasLLM(ollama_llm)
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        # Use only metrics that don't require ground truth/reference
        self.metrics = [answer_relevancy]
        self.model_server = model_server
        self.model_name = model_name
        self.debug_mode = debug_mode

    def load_and_split_docs(self, data_dir: str, filepath: str = None):
        """
        Loads documents and splits them using the TextSplitter from the main RAG app.
        """
        
        if filepath:
            splitter = UniversalDocumentSplitter(filepath=filepath)
        elif data_dir:
            splitter = UniversalDocumentSplitter(data_dir=data_dir)
        else:
            splitter = UniversalDocumentSplitter(filepath=settings.DATA_FILEPATH)

        docs = splitter.split_and_process()
        return docs

    def evaluate_rag_system(self, question: str, rag_response: str, docs: list):
        """Evaluates the RAG system's response using Ragas and returns the scores."""

        if docs:
            # If docs are provided, use them as context
            contexts = [[doc.page_content for doc in docs]]
        else:
            # If no docs, provide an empty list for contexts
            contexts = [[]]

        data = {
            "question": [question],
            "answer": [rag_response],
            "contexts": contexts,
        }

        if self.debug_mode:
            st.markdown("**Debug: Input Data (data):**")
            st.write(data)

        dataset = Dataset.from_dict(data)

        result = evaluate(
            dataset=dataset,
            metrics=self.metrics,
            llm=self.ragas_llm,
            embeddings=self.embeddings
        )

        if self.debug_mode:
            st.markdown("**Debug: Raw Evaluation Result (result):**")
            st.write(result)

        ragas_answer_relevancy = result["answer_relevancy"]
        #ragas_faithfulness = result["faithfulness"]

        return {
            "answer_relevancy": ragas_answer_relevancy,
            #"faithfulness": ragas_faithfulness,
        }
