from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_gigachat.chat_models import GigaChat
from langchain_community.document_loaders.csv_loader import CSVLoader

from src.core.config import settings
from src.retriever.splitter import UniversalDocumentSplitter, CSVSplitter
from src.utils.loaders import load_embedding_model

class RecommendationService:
    def __init__(self, retriever, prompt_template, catalog_retriever):
        self.retriever = retriever
        self.prompt_template = prompt_template
        self.catalog_retriever = catalog_retriever
        self.rag_chain = None

    def _create_rag_chain(self, model_server, model_name):
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
        
        # Retrieve both context and catalog
        rag_chain_with_catalog = RunnableParallel(
            {
                "context": self.retriever,
                "question": RunnablePassthrough(),
                "catalog": self.catalog_retriever,
            }
        ).assign(
            context=lambda x: "\n".join(
                [
                    "## Описание:\n" + dc.page_content
                    for dc in x["context"]
                ]
                + ["## Каталог вин:\n" + "".join(
                    [doc.page_content for doc in x["catalog"]]
                    )
                ]
            )
        )

        self.rag_chain = (
            rag_chain_with_catalog
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
    # Create splitter for the main data, passing name_column to CSVSplitter
    splitter = UniversalDocumentSplitter(
        filepath=settings.DATA_FILEPATH,
        data_dir=settings.RAG_DATA_DIR,
        chunk_size=500,
        chunk_overlap=50,
        name_column=settings.WINE_NAME_COLUMN
    )

    docs = splitter.split_and_process()
    embedding_model = load_embedding_model()
    retriever = splitter.create_vector_store(docs, embedding_model)

    # Create catalog retriever (no need for name_column here)
    catalog_splitter = UniversalDocumentSplitter(
        filepath=settings.CATALOGUE_FILEPATH,
        chunk_size=3000,
        chunk_overlap=0
    )
    catalog_docs = catalog_splitter.split_and_process()
    catalog_retriever = catalog_splitter.create_vector_store(
        catalog_docs, embedding_model
    )
    
    prompt = PromptTemplate.from_template(settings.PROMPT_TEMPLATE)

    return RecommendationService(retriever, prompt, catalog_retriever)