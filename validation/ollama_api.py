import httpx
import time
from langchain_ollama.llms import OllamaLLM
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from config import OLLAMA_API_BASE
from typing import Optional


class OllamaAPIWrapper:
    """Wrapper for interacting with the Ollama API."""

    def __init__(self, base_url: str = OLLAMA_API_BASE, model: str = "gemma2:2b-instruct-fp16", temperature: float = 0):
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        self.llm = self._initialize_llm(model, temperature)

    def _initialize_llm(self, model: str, temperature: float) -> OllamaLLM:
        """Initializes the Ollama LLM."""
        return OllamaLLM(
            base_url=self.base_url,
            model=model,
            callback_manager=self.callback_manager,
            temperature=temperature
        )
    
    def update_llm(self, model: Optional[str] = None, temperature: Optional[float] = None):
        """Updates the Ollama LLM model and/or temperature"""
        if model is not None:
            self.model = model
        if temperature is not None:
            self.temperature = temperature
        self.llm = self._initialize_llm(self.model, self.temperature)

    async def get_models(self):
        """Retrieves available models from the Ollama API."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/api/tags")
            return response.json()

    def generate_summary(self, text: str) -> tuple[str, float, int, int, float]:
        """Generates a summary using Ollama and returns it along with metrics."""

        start_time = time.time()
        summary = self.llm.invoke(f"Please summarize the following text:\n\n{text}")
        end_time = time.time()

        processing_time = end_time - start_time
        input_tokens = len(text.split())
        output_tokens = len(summary.split())
        throughput = output_tokens / processing_time

        return summary, processing_time, input_tokens, output_tokens, throughput
