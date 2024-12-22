from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.retriever.semantic_splitter import SemanticSplitter
import os


class CSVSplitter:
    """Splits CSV files into chunks based on a specified chunk size."""

    def __init__(self, filepath, source_column, chunk_size=500, chunk_overlap=0):
        self.filepath = filepath
        self.source_column = source_column
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _load_data(self):
        loader = CSVLoader(self.filepath, source_column=self.source_column, encoding="utf-8")
        return loader.load()

    def split_and_process(self):
        docs = self._load_data()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        split_docs = text_splitter.split_documents(docs)
        return split_docs


class UniversalDocumentSplitter:
    """
    Splits documents based on their file type.
    Uses CSVSplitter for .csv files and SemanticSplitter for other types.
    """

    def __init__(self, filepath, source_column=None, chunk_size=500, chunk_overlap=50, data_dir=None):
        self.filepath = filepath
        self.source_column = source_column
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.data_dir = data_dir  # Optional directory for multiple files

        self.splitter = self._get_splitter()

    def _get_splitter(self):
        if self.filepath and self.filepath.lower().endswith(".csv"):
            return CSVSplitter(
                self.filepath,
                self.source_column,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        else:
            return TextSplitter(
                filepath=self.filepath,
                source_column=self.source_column,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )

    def _load_data(self):
        """Loads data based on file type and whether a directory is provided."""
        if self.data_dir:
            docs = []
            for filename in os.listdir(self.data_dir):
                filepath = os.path.join(self.data_dir, filename)
                if filename.lower().endswith(".csv"):
                    loader = CSVLoader(filepath, source_column=self.source_column, encoding="utf-8")
                elif filename.lower().endswith(".pdf"):
                    loader = PyPDFLoader(filepath)
                else:
                    print(f"Unsupported file type: {filename}. Skipping.")
                    continue
                docs.extend(loader.load())
            return docs
        elif self.filepath:
            if self.filepath.lower().endswith(".csv"):
                loader = CSVLoader(self.filepath, source_column=self.source_column, encoding="utf-8")
            elif self.filepath.lower().endswith(".pdf"):
                loader = PyPDFLoader(self.filepath)
            else:
                raise ValueError("Unsupported file type. Only CSV and PDF are supported.")
            return loader.load()
        else:
            raise ValueError("Either 'filepath' or 'data_dir' must be provided.")

    def split_and_process(self):
        docs = self.splitter._load_data()
        return self.splitter.split_and_process()

    def create_vector_store(self, documents, embeddings):
        db = Chroma.from_documents(documents, embeddings)
        return db.as_retriever()


class TextSplitter:
    def __init__(self, filepath, source_column, chunk_size=500, chunk_overlap=50):
        self.filepath = filepath
        self.source_column = source_column
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = self._create_splitter()

    def _create_splitter(self):
        splitter_model = SentenceTransformer("TaylorAI/bge-micro-v2")
        splitter = SemanticSplitter(splitter_model, buffer_back=1, buffer_forward=0, threshold=85)
        return splitter

    def _load_data(self):
        loader = DirectoryLoader(
            self.filepath,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True,
            use_multithreading=True
        )
        return loader.load()

    def split_and_process(self):
        docs = self._load_data()
        processed_docs = []
        for doc in docs:
            chunks = self.splitter.split(doc.page_content)
            if not chunks:
                print("Warning: No chunks generated for this document. Adding the entire document as a single chunk.")
                processed_docs.append(doc)
            else:
                for chunk in chunks:
                    processed_docs.append(type(doc)(page_content=chunk, metadata=doc.metadata))
        return processed_docs

    def create_vector_store(self, documents, embeddings):
        db = Chroma.from_documents(documents, embeddings)
        return db.as_retriever()
