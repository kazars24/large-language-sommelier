from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_gigachat.chat_models import GigaChat

from src.core.config import settings
from src.retriever.splitter import UniversalDocumentSplitter
from src.utils.loaders import load_embedding_model


class RecommendationService:
    def __init__(self, retriever, prompt_template):
        self.retriever = retriever
        self.prompt_template = prompt_template
        self.rag_chain = None  # Initialize rag_chain to None

    def _create_rag_chain(self, model_server, model_name):
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        if model_server == "ollama":
            llm = OllamaLLM(
                base_url=settings.OLLAMA_API_BASE,
                model=model_name,
                timeout=30
            )
            llm.verbose = False
        elif model_server == "gigachat":
            llm = GigaChat(
                credentials=settings.GIGACHAT_CREDENTIALS,
                model=model_name,
                timeout=30,
                verify_ssl_certs=False
            )
            llm.verbose = False
        else:
            raise ValueError(f"Invalid model server: {model_server}")

        self.rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt_template
            | llm
            | StrOutputParser()
        )

    def get_recommendation(self, query: str, model_server: str, model_name: str):
        print(f"RecommendationService received:")
        print(f"  Query: {query}")
        print(f"  Model Server: {model_server}")
        print(f"  Model Name: {model_name}")

        if self.rag_chain is None:
            self._create_rag_chain(model_server, model_name)
        elif self.rag_chain.steps[2].model != model_name:
            self._create_rag_chain(model_server, model_name)
        return self.rag_chain.invoke(query)


def get_recommendation_service():
    splitter = UniversalDocumentSplitter(
        filepath=settings.DATA_FILEPATH,
        source_column=settings.DATA_SOURCE_COLUMN,
        chunk_size=500,
        chunk_overlap=50
    )
    docs = splitter.split_and_process()
    embedding_model = load_embedding_model()
    retriever = splitter.create_vector_store(docs, embedding_model)
    prompt = PromptTemplate.from_template(settings.PROMPT_TEMPLATE)
    return RecommendationService(retriever, prompt)
