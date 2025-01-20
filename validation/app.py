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
from src.core.config import settings as main_rag_settings

langfuse = setup_langfuse()

API_URL = "http://localhost:8000"

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
        return response.json()
    else:
        st.error(f"Error from RAG API: {response.status_code} - {response.text}")
        return None

def update_rag_data(filepath, data_dir):
    if filepath:
        # If a file is uploaded, send it as a single file
        response = requests.post(
            f"{API_URL}/update_data/",
            params={"filepath": filepath}
        )
    elif data_dir:
        # If a directory is provided, send the directory path
        response = requests.post(
            f"{API_URL}/update_data/",
            params={"data_dir": data_dir}
        )
    else:
        # If neither file nor directory is provided, use default settings
        response = requests.post(f"{API_URL}/update_data/")

    if response.status_code != 200:
        st.error(f"Failed to update RAG data: {response.text}")
        return False
    return True

def load_and_split_docs(data_dir, filepath):
    splitter = UniversalDocumentSplitter(
        filepath=filepath,
        data_dir=data_dir,
    )
    docs = splitter.split_and_process()
    return docs

def main():
    st.title("RAG System Validation")

    model_server = st.sidebar.selectbox("Select Model Server", ["ollama", "gigachat"], index=0)

    local_models = fetch_local_models()
    if model_server == "ollama":
        selected_model = st.sidebar.selectbox("Select Ollama Model", local_models, index=0)
    else:
        selected_model = st.sidebar.selectbox("Select Model", ["GigaChat"], index=0)

    critic_model = st.sidebar.selectbox("Select Critic Model (Ollama)", local_models, index=0)

    question = st.text_area("Enter your question about wine:")
    data_dir = st.text_input("Enter the directory containing your RAG documents (or leave blank for default wine data):", "")

    uploaded_file = st.file_uploader("Choose a CSV or PDF file", type=["csv", "pdf"])
    temp_file_path = None  # Initialize temp_file_path here

    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        temp_file_path = os.path.join("temp", uploaded_file.name)
        os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    if st.button("Update RAG Data"):
        # Pass temp_file_path to update_rag_data only if a file was uploaded
        if update_rag_data(temp_file_path, data_dir):
            st.success("RAG data updated successfully.")
        else:
            st.error("Failed to update RAG data.")

    default_system_prompt = main_rag_settings.PROMPT_TEMPLATE
    system_prompt = st.text_area("Edit System Prompt (Optional):", default_system_prompt, height=200)

    default_embedding_model = main_rag_settings.EMBEDDING_MODEL_NAME
    embedding_model_name = st.text_input("Override Embedding Model Name (Optional):", default_embedding_model)

    debug_mode = st.checkbox("Enable Debug Mode")

    if st.button("Evaluate RAG System"):
        if question:
            # Update system prompt in the main RAG app's settings
            main_rag_settings.PROMPT_TEMPLATE = system_prompt
            trace = langfuse.trace(name=f"wine-recommendation-debug")

            span = trace.span(name="generate_summary", input = {'text': question})
            response = get_rag_recommendation(question, model_server, selected_model)
            rag_response = response["response"]
            span.end(
                output=response
            )

            if rag_response:
                st.markdown("**RAG System Response:**")
                st.write(rag_response)

                # Load and split documents using UniversalDocumentSplitter
                docs = load_and_split_docs(data_dir, temp_file_path)

                with st.spinner("Evaluating with Ragas..."):
                    ollama_wrapper = OllamaAPIWrapper(base_url=OLLAMA_API_BASE)
                    ollama_wrapper.update_llm(model=critic_model)

                    evaluator = RagasEvaluator(ollama_wrapper.llm, model_server, selected_model, embedding_model_name, debug_mode)
                    span = trace.span(name="evaluate_summary", input = {'text': question})
                    
                    eval_result = evaluator.evaluate_rag_system(question, rag_response, docs)
                    span.end(
                        output=eval_result
                    )

                    # Replace the existing metrics display code with:
                    st.markdown("**Evaluation Results:**")

                    # Create columns for metrics
                    col1, col2, col3, col4, col5 = st.columns(5)

                    # Display each metric in a separate column
                    col1.metric("Answer Relevancy", f"{eval_result['answer_relevancy']:.4f}")
                    #col2.metric("Context Precision", f"{eval_result['context_precision']:.4f}")
                    #col3.metric("Faithfulness", f"{eval_result['faithfulness']:.4f}")
                    col4.metric("Answer Completeness", f"{eval_result['answer_completeness']:.4f}")
                    col5.metric("Harmfulness", f"{eval_result['harmfulness']:.4f}")

                    if debug_mode:
                        st.markdown("### Metric Explanations")
                        st.markdown("""
                        - **Answer Relevancy**: Measures how well the answer addresses the question.
                        - **Context Precision**: Evaluates if the retrieved contexts are necessary and sufficient for answering the question.
                        - **Faithfulness**: Measures if the generated answer is faithful to the provided context.
                        - **Answer Completeness**: Evaluates if the answer fully addresses all parts of the question.
                        - **Harmfulness**: Determines if the response contains harmful, biased, or toxic content.
                        """)

        else:
            st.warning("Please enter a question to evaluate.")

if __name__ == "__main__":
    main()
