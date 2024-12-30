from ragas import evaluate
from ragas.metrics import (
    answer_relevancy
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.outputs import LLMResult
from datasets import Dataset
from typing import List, Dict, Any
from ragas.llms.base import BaseRagasLLM
from langchain_ollama.llms import OllamaLLM
import streamlit as st
import asyncio
from functools import partial

class OllamaRagasLLM(BaseRagasLLM):
    """Enhanced Ragas LLM implementation for Ollama with better error handling and timeouts."""

    def __init__(self, ollama_instance: OllamaLLM, timeout: int = 30):
        self.ollama = ollama_instance
        self.timeout = timeout

    async def _generate_with_timeout(self, prompt: str) -> str:
        """Generate text with a timeout to prevent hanging."""
        try:
            response = await asyncio.wait_for(
                self.ollama.ainvoke(prompt),
                timeout=self.timeout
            )
            return response
        except asyncio.TimeoutError:
            return "Response generation timed out"
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def generate_text(self, prompt: str, **kwargs) -> str:
        """Synchronous generation with timeout."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        response = loop.run_until_complete(self._generate_with_timeout(prompt))
        loop.close()
        return response

    async def agenerate_text(self, prompt: str, **kwargs) -> str:
        """Asynchronous generation with timeout."""
        return await self._generate_with_timeout(prompt)

    async def generate(
        self,
        prompts: List[str],
        n: int = 1,
        temperature: float = 0,
        **kwargs
    ) -> LLMResult:
        """Batch generation with parallel processing and timeouts."""
        tasks = [self._generate_with_timeout(prompt) for prompt in prompts]
        responses = await asyncio.gather(*tasks)
        generations = [[{"text": response, "output": response}] for response in responses]
        return LLMResult(generations=generations)

class RagasEvaluator:
    """Enhanced class for evaluating RAG systems using multiple metrics."""

    def __init__(
        self,
        ollama_llm: OllamaLLM,
        model_server: str,
        model_name: str,
        embedding_model_name: str,
        debug_mode: bool = False,
        timeout: int = 30
    ):
        self.ragas_llm = OllamaRagasLLM(ollama_llm, timeout=timeout)
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.metrics = [
            answer_relevancy,    # Measures if the answer is relevant to the question
        ]
        self.model_server = model_server
        self.model_name = model_name
        self.debug_mode = debug_mode

    def _prepare_evaluation_data(
        self,
        question: str,
        rag_response: str,
        docs: list
    ) -> Dict[str, List[Any]]:
        """Prepares data for evaluation in the required format."""
        contexts = [doc.page_content for doc in docs] if docs else [""]
        
        data = {
            "question": [question],
            "answer": [rag_response],
            "contexts": [contexts],
            "ground_truth": [""],  # Empty ground truth as we don't have it
        }

        if self.debug_mode:
            st.markdown("**Debug: Prepared Evaluation Data:**")
            st.json(data)

        return data

    async def evaluate_rag_system_async(
        self,
        question: str,
        rag_response: str,
        docs: list
    ) -> Dict[str, float]:
        """Asynchronous evaluation of the RAG system."""
        data = self._prepare_evaluation_data(question, rag_response, docs)
        dataset = Dataset.from_dict(data)

        result = await asyncio.wait_for(
            evaluate(
                dataset=dataset,
                metrics=self.metrics,
                llm=self.ragas_llm,
                embeddings=self.embeddings,
                batch_size=16,
            ),
            timeout=240  # Overall evaluation timeout
        )

        if self.debug_mode:
            st.markdown("**Debug: Raw Evaluation Result:**")
            st.write(result)

        return {
            "answer_relevancy": float(result["answer_relevancy"][0])
        }

    def evaluate_rag_system(
        self,
        question: str,
        rag_response: str,
        docs: list
    ) -> Dict[str, float]:
        """Synchronous wrapper for RAG system evaluation."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                self.evaluate_rag_system_async(question, rag_response, docs)
            )
            return result
        finally:
            loop.close()
