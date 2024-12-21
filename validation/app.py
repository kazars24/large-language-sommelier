import streamlit as st
import requests

API_URL = "http://localhost:8000"

def fetch_models():
    response = requests.get(f"{API_URL}/models")
    return [model["name"] for model in response.json()["models"]]

def summarize_text(text, model):
    response = requests.post(
        f"{API_URL}/summarize",
        json={"text": text, "model": model}
    )
    return response.json()

def main():
    st.title("Text Summarization")

    # Model selection
    models = fetch_models()
    selected_model = st.selectbox("Select Model", models)

    # Input text
    input_text = st.text_area("Input Text", height=200)

    if st.button("Summarize"):
        if input_text:
            with st.spinner("Generating summary..."):
                result = summarize_text(input_text, selected_model)
                
                # Display summary
                st.subheader("Summary")
                st.text_area("", result["summary"], height=100)
                
                # Display metrics
                st.subheader("Metrics")
                metrics = result["metrics"]
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Throughput (tokens/s)", 
                             f"{metrics['throughput']:.2f}")
                with col2:
                    st.metric("Faithfulness Score", 
                             f"{metrics['faithfulness_score']:.2f}")
                with col3:
                    st.metric("Relevancy Score", 
                             f"{metrics['relevancy_score']:.2f}")
        else:
            st.warning("Please enter some text to summarize")

if __name__ == "__main__":
    main()
