# RAG System Validation Application

This application provides a Streamlit-based interface for evaluating the performance and quality of a Retrieval-Augmented Generation (RAG) system. It leverages various tools, including Langfuse for tracing, Ollama for local large language model (LLM) serving, and Ragas for advanced RAG system evaluation.

## Features

*   **Model Selection:** Choose between different LLMs served via Ollama or GigaChat for generating responses.
*   **Critic Model:** Specify a separate critic model (from Ollama) to provide feedback on the generated responses.
*   **RAG Data Update:** Update the RAG system's knowledge base by uploading CSV or PDF files or providing a directory of data.
*   **System Prompt Customization:** Modify the system prompt that guides the RAG system's behavior.
*   **Embedding Model Override:** Optionally, change the embedding model used by Ragas for evaluating semantic similarity.
*   **Debug Mode:** Enables detailed logging and displays intermediate data for debugging the evaluation process.
*   **Ragas Evaluation:** Integrates the Ragas framework to evaluate the RAG system based on metrics such as:
    *   **Answer Relevancy:** Measures how well the generated answer addresses the given question.
    *   **Answer Completeness:**  Evaluates if the answer fully addresses all parts of the question.
    *   **Harmfulness:** Determines if the response contains harmful, biased, or toxic content.

## Project Structure

The project is organized into the following files and directories:

*   **`app.py`:** The main Streamlit application that provides the user interface and orchestrates the RAG system evaluation.
*   **`langfuse_setup.py`:**  Sets up the Langfuse client for tracing and monitoring.
*   **`ollama_api.py`:**  Provides a wrapper for interacting with the Ollama API to fetch available models and generate text summaries.
*   **`models.py`:** Defines Pydantic models for request and response data structures.
*   **`ragas_evaluation.py`:** Implements the Ragas evaluation logic, including custom metrics and integration with LangChain.
*   **`config.py`:** Contains configuration settings for Langfuse, Ollama, and HuggingFace embeddings.
*   **`validation/`:** (Likely intended, but currently missing) This directory would likely contain modules related to other aspects of validation, potentially including the `UniversalDocumentSplitter` mentioned in `app.py`. You might need to create it and move related code there for better organization.

## Setup and Installation

### Prerequisites

1. **Python:** Ensure you have Python 3.9 or later installed.
2. **Ollama:** Install and run Ollama to serve local LLMs. You can download it from [https://ollama.ai/](https://ollama.ai/).
3. **Langfuse:** Create a free account and obtain your API keys.
4. **GigaChat API:** If you want to use GigaChat, you will need access to its API.

### Installation Steps

1. **Clone the repository:**

    ```bash
    git clone <your-repository-url>
    cd <repository-name>
    ```

2. **Create a virtual environment (recommended):**

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Linux/macOS
    .venv\Scripts\activate  # On Windows
    ```

3. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```
    (Make sure you create a `requirements.txt` file with all the necessary packages listed)

4. **Environment Variables:**
    *   Create a `.env` file in the project's root directory.
    *   Set the following environment variables in the `.env` file:

        ```
        LANGFUSE_PUBLIC_KEY=<your-langfuse-public-key>
        LANGFUSE_SECRET_KEY=<your-langfuse-secret-key>
        LANGFUSE_HOST=http://localhost:3000 # Or your Langfuse instance URL
        OLLAMA_API_BASE=http://<your-ollama-server-ip>:11434
        ```

5. **Pull LLMs in Ollama:**

    ```bash
    ollama pull gemma:2b-instruct-fp16 # Or any other model you want to use
    ```

### Create temporary folder
Create folder `temp` in project's root directory. This folder will be used to store temporary files.

## Running the Application

1. **Start the RAG service:**
    Ensure that the RAG service you are evaluating is running and accessible at `http://localhost:8000` (or the URL you have configured in `app.py`). The RAG service is assumed to have the following endpoints:
    * `/recommend/` for generating recommendations.
    * `/update_data/` for updating RAG data.

2. **Launch the Streamlit app:**

    ```bash
    streamlit run app.py
    ```

3. **Access the app:**
    Open your web browser and go to the URL provided by Streamlit (usually `http://localhost:8501`).

## Usage

1. **Select Model Server:** Choose whether to use Ollama or GigaChat for serving the main RAG model.
2. **Select Model:** Pick the specific model from the available options (models pulled in Ollama or GigaChat).
3. **Select Critic Model:** Choose an Ollama model to act as a critic for evaluation.
4. **Enter Question:** Type the question you want to ask the RAG system.
5. **Update RAG Data (Optional):**
    *   Provide a directory containing your data files, or
    *   Upload a CSV or PDF file.
    *   Click "Update RAG Data" to update the system's knowledge.
6. **Edit System Prompt (Optional):** Modify the system prompt to influence the RAG system's response generation.
7. **Override Embedding Model (Optional):** Change the embedding model used by Ragas.
8. **Enable Debug Mode (Optional):** Turn on debug mode for more detailed output.
9. **Evaluate RAG System:** Click the "Evaluate RAG System" button.

The application will then:

*   Send the question to the RAG system.
*   Display the RAG system's response.
*   Evaluate the response using Ragas and the critic model.
*   Show the evaluation metrics (Answer Relevancy, Answer Completeness, and Harmfulness).

## Troubleshooting

*   **Ollama not running:** Make sure Ollama is installed and running before starting the application.
*   **Model not found:** If you encounter an error about a model not being found, ensure you have pulled the model in Ollama using `ollama pull <model_name>`.
*   **RAG service not accessible:** Verify that your RAG service is running and that the API URL in `app.py` is correct.
*   **Langfuse errors:** Double-check your Langfuse API keys and host in the `.env` file.
*   **Dependency issues:** If you have problems with dependencies, try deleting your virtual environment and reinstalling the packages.

## Further Development

*   **Add More Ragas Metrics:** Integrate other Ragas metrics like `context_precision` and `faithfulness` to provide a more comprehensive evaluation.
*   **Support for Other LLM Providers:** Extend the application to support other LLM providers besides Ollama and GigaChat.
*   **Batch Evaluation:** Implement functionality to evaluate the RAG system on a dataset of questions.
*   **User Interface Improvements:** Enhance the Streamlit UI to make it more user-friendly and visually appealing.
*   **Reporting:** Generate reports summarizing the evaluation results, including charts and graphs.
*   **Integration with CI/CD:** Automate the RAG system evaluation as part of a continuous integration/continuous deployment pipeline.

This detailed `README.md` should provide a solid foundation for understanding, using, and extending your RAG validation application. Remember to fill in any missing parts, like the `validation/` directory structure, and update the `requirements.txt` file with the required packages.