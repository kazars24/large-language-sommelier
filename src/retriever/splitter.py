from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from typing import List, Optional, Any
import os


class UniversalDocumentSplitter:
    """
    Universal document splitter that handles both CSV and PDF files,
    using different processing strategies for each.
    """

    def __init__(
        self,
        filepath: Optional[str] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        data_dir: Optional[str] = None
    ):
        self.filepath = filepath
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def _load_file(self, filepath: str) -> List:
        """Loads a single file with appropriate processing."""
        if str(filepath).lower().endswith(".csv"):
            return CSVLoader(filepath, encoding="utf-8").load()
        elif str(filepath).lower().endswith(".pdf"):
            docs = PyPDFLoader(filepath).load()
            return self.text_splitter.split_documents(docs)
        else:
            raise ValueError("Unsupported file type. Only CSV and PDF are supported.")

    def split_and_process(self) -> List:
        """Main method to load and process documents."""
        if self.data_dir:
            docs = []
            for filename in os.listdir(self.data_dir):
                filepath = os.path.join(self.data_dir, filename)
                if filename.lower().endswith((".csv", ".pdf")):
                    try:
                        docs.extend(self._load_file(filepath))
                    except Exception as e:
                        print(f"Error processing {filename}: {str(e)}")
                else:
                    print(f"Skipping unsupported file: {filename}")
            return docs
        elif self.filepath:
            return self._load_file(self.filepath)
        else:
            raise ValueError("Either 'filepath' or 'data_dir' must be provided.")

    def create_vector_store(self, documents: List, embeddings) -> Any:
        """Creates a FAISS vector store from the documents."""
        return FAISS.from_documents(documents, embeddings).as_retriever()
