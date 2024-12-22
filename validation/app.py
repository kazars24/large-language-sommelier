import streamlit as st
import requests
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from validation.ragas_evaluation import RagasEvaluator
from validation.ollama_api import OllamaAPIWrapper
from validation.config import OLLAMA_API_BASE
from src.retriever.splitter import UniversalDocumentSplitter
from langfuse_setup import setup_langfuse

langfuse = setup_langfuse()

API_URL = "http://localhost:8000"


def fetch_models(model_server):
    response = requests.get(f"{API_URL}/models", params={"model_server": model_server})
    response.raise_for_status()
    return [model["name"] for model in response.json()["models"]]


def fetch_local_models():
    """Fetches models from the local Ollama server."""
    ollama_wrapper = OllamaAPIWrapper(base_url=OLLAMA_API_BASE)
    models = ollama_wrapper.get_models()
    return [model["name"] for model in models.get("models", [])]


def get_rag_recommendation(question, model_server, model_name):
    response = requests.post(
        f"{API_URL}/recommend/",
        json={
            "query": {"question": question},
            "model_choice": {
                "model_server": model_server,
                "model_name": model_name
            }
        }
    )
    if response.status_code == 200:
        return response.json()["response"]
    else:
        st.error(f"Error from RAG API: {response.status_code} - {response.text}")
        return None


def update_rag_data(filepath, source_column):
    response = requests.post(
        f"{API_URL}/update_data/",
        params={"filepath": filepath, "source_column": source_column}
    )
    if response.status_code != 200:
        st.error(f"Failed to update RAG data: {response.text}")
        return False
    return True


def load_and_split_docs(data_dir, filepath, source_column):
    splitter = UniversalDocumentSplitter(
        filepath=filepath,
        source_column=source_column,
        data_dir=data_dir,
    )
    docs = splitter.split_and_process()
    return docs


def main():
    st.title("RAG System Validation")

    model_server = st.sidebar.selectbox("Select Model Server", ["ollama", "gigachat"], index=0)

    if model_server == "ollama":
        local_models = fetch_local_models()
        selected_model = st.sidebar.selectbox("Select Ollama Model", local_models, index=0)
    else:
        selected_model = st.sidebar.selectbox("Select Model", ["GigaChat"], index=0)

    critic_model = st.sidebar.selectbox("Select Critic Model (Ollama)", local_models, index=0)

    question = st.text_area("Enter your question about wine:")
    data_dir = st.text_input("Enter the directory containing your RAG documents (or leave blank for default wine data):", "")

    uploaded_file = st.file_uploader("Choose a CSV or PDF file", type=["csv", "pdf"])

    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        temp_file_path = os.path.join("temp", uploaded_file.name)
        os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Determine file type and set parameters for data update
        if uploaded_file.type == "text/csv":
            source_column = st.text_input("Enter the source column name for the CSV:", "text")
            filepath = temp_file_path
        elif uploaded_file.type == "application/pdf":
            source_column = ""  # Not applicable for PDFs
            filepath = temp_file_path
        else:
            st.error("Unsupported file type. Please upload a CSV or PDF.")
            filepath = None
            source_column = None

        if st.button("Update RAG Data"):
            if filepath and update_rag_data(filepath, source_column):
                st.success("RAG data updated successfully.")
            else:
                st.error("Failed to update RAG data.")

    if st.button("Evaluate RAG System"):
        if question:
            rag_response = get_rag_recommendation(question, model_server, selected_model)

            if rag_response:
                st.markdown("**RAG System Response:**")
                st.write(rag_response)

                # Load and split documents using UniversalDocumentSplitter
                if uploaded_file is not None:
                    docs = load_and_split_docs(None, filepath, source_column)  # Pass None for data_dir when using uploaded_file
                elif data_dir:
                    docs = load_and_split_docs(data_dir, None, None)  # Pass None for filepath and source_column when using data_dir
                else:
                    docs = []

                with st.spinner("Evaluating with Ragas..."):
                    ollama_wrapper = OllamaAPIWrapper(base_url=OLLAMA_API_BASE)
                    ollama_wrapper.update_llm(model=critic_model)

                    evaluator = RagasEvaluator(ollama_wrapper.llm, model_server, selected_model)
                    
                    eval_result = evaluator.evaluate_rag_system(question, rag_response, docs)

                    st.markdown("**Evaluation Results:**")
                    col1, col2 = st.columns(2)

                    col2.metric("Ragas Answer Relevancy", f"{eval_result['ragas_answer_relevancy'][0]:.4f}")

        else:
            st.warning("Please enter a question to evaluate.")

if __name__ == "__main__":
    main()
