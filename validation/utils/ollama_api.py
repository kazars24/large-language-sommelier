from typing import Optional, Tuple, List, Union, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.utils import trimAndLoadJson
from langchain_community.callbacks import get_openai_callback
from langchain_ollama.llms import OllamaLLM


class CustomChatOpenAI(ChatOpenAI):
    format: str = None

    def __init__(self, format: str = None, **kwargs):
        super().__init__(**kwargs)
        self.format = format

    async def _acreate(
        self, messages: List[BaseMessage], **kwargs
    ) -> ChatResult:
        if self.format:
            kwargs["format"] = self.format
        return await super()._acreate(messages, **kwargs)


class OllamaEvalLLM(DeepEvalBaseLLM):
    def __init__(
        self,
        model_name: str,
        base_url: str = "http://192.168.1.198:11434/",
        json_mode: bool = True,
        temperature: float = 0,
        *args,
        **kwargs
    ):

        self.model_name = model_name
        self.base_url = base_url
        self.json_mode = json_mode
        self.temperature = temperature
        self.args = args
        self.kwargs = kwargs
        super().__init__(model_name)

    def load_model(self):
        """Load and configure the Ollama model."""
        return OllamaLLM(
            base_url=self.base_url,
            model=self.model_name,
            temperature=0,
            format='json',
            num_ctx=32768
        )

    def generate(self, prompt: str, schema: Any = None) -> Union[Any, Tuple[str, float]]:

        chat_model = self.load_model()
        with get_openai_callback() as cb:
            res = chat_model.invoke(prompt)
            print(f'ollama generate: {res}')
            if schema is not None:
                try:
                    # Try to parse the response using the schema
                    data = trimAndLoadJson(res, None)
                    print(f'ollama trimAndLoadJson: {data}')
                    return schema(**data)
                except Exception:
                    # If schema parsing fails, return parsed JSON
                    return trimAndLoadJson(res, None)
            return res, 0.0

    async def a_generate(self, prompt: str, schema: Any = None) -> Union[Any, Tuple[str, float]]:

        chat_model = self.load_model()
        with get_openai_callback() as cb:
            res = await chat_model.ainvoke(prompt)
            print(f'ollama a_generate: {res}')
            if schema is not None:
                try:
                    # Try to parse the response using the schema
                    data = trimAndLoadJson(res, None)
                    print(f'ollama trimAndLoadJson: {data}')
                    return schema(**data)
                except Exception:
                    # If schema parsing fails, return parsed JSON
                    return trimAndLoadJson(res, None)
            return res, 0.0

    def get_model_name(self) -> str:
        """Get the name of the current model."""
        return self.model_name
