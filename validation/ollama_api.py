import httpx
import time
from langchain_community.llms.ollama import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from config import OLLAMA_API_BASE
from typing import Optional, Dict, Any

class OllamaAPIWrapper:
    """Wrapper for interacting with the Ollama API."""

    def __init__(self, base_url: str = OLLAMA_API_BASE, model: str = "gemma:2b-instruct-fp16", temperature: float = 0):
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        self.llm = self._initialize_llm(model, temperature)

    def _initialize_llm(self, model: str, temperature: float) -> Ollama:
        """Initializes the Ollama LLM."""
        return Ollama(
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

    def get_models(self) -> Dict[str, Any]:
        """Retrieves available models from the Ollama API."""
        try:
            with httpx.Client() as client:
                response = client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()  # Raise an exception for bad status codes
                return response.json()
        except httpx.HTTPError as e:
            print(f"HTTP error occurred: {e}")
            return {"models": []}  # Return an empty list of models in case of an error
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return {"models": []}

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
