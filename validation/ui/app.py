import sys
import os
import streamlit as st
import asyncio
from pathlib import Path
import json
from datetime import datetime
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.metrics import ValidationMetricsCollector
from core.tester import RAGSystemTester
from langfuse import Langfuse
import pandas as pd
import httpx

# Initialize Langfuse client
langfuse = Langfuse(
    public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.environ.get("LANGFUSE_SECRET_KEY"),
    host='http://localhost:3000'
)

# Initialize metrics collector
metrics_collector = ValidationMetricsCollector(langfuse)

def load_test_queries():
    """Load sample test queries"""
    return [
        "Порекомендуй красное вино стоимостью не более 20000",
        "Какое вино подойдет к рыбе?",
        "Порекомендуй сладкое десертное вино",
        "Вино с нотками персика"
    ]

async def get_ollama_models(base_url: str = "http://192.168.1.198:11434"):
    """Get available Ollama models asynchronously"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{base_url}/api/tags")
            response.raise_for_status()
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
    except Exception as e:
        st.error(f"Error fetching Ollama models: {e}")
        return ["qwen2.5:7b-instruct-q4_0"]  # Default fallback model

def main():
    st.title("Wine Recommendation RAG System Validator")
    
    # Initialize session state for models
    if 'ollama_models' not in st.session_state:
        st.session_state.ollama_models = asyncio.run(get_ollama_models())
    
    # Sidebar configuration
    st.sidebar.header("Model Configuration")
    
    # Model server selection
    model_server = st.sidebar.selectbox(
        "Model Server",
        ["ollama", "gigachat"],
        index=0
    )
    
    # Model selection based on server
    if model_server == "ollama":
        # RAG system model
        rag_model = st.sidebar.selectbox(
            "RAG System Model",
            st.session_state.ollama_models,
            index=0,
            key="rag_model"
        )
        
        # Evaluation model (can be different from RAG model)
        eval_model = st.sidebar.selectbox(
            "Evaluation Model",
            st.session_state.ollama_models,
            index=0,
            key="eval_model"
        )
    else:
        rag_model = eval_model = st.sidebar.selectbox(
            "Model Name",
            ["GigaChat:latest"],
            index=0
        )
    
    # Data source configuration
    st.sidebar.header("Data Source")
    data_source_type = st.sidebar.radio(
        "Data Source Type",
        ["Single File", "Directory"]
    )
    
    if data_source_type == "Single File":
        data_path = st.sidebar.text_input("File Path", "I:\itmo\large-language-sommelier\data\wine_desc.csv")
    else:
        data_path = st.sidebar.text_input("Directory Path", "data/")
    
    # Testing section
    st.header("Testing")
    
    # Test queries input
    test_queries = st.text_area(
        "Test Queries (one per line)",
        value="\n".join(load_test_queries()),
        height=200
    )
    queries = [q.strip() for q in test_queries.split("\n") if q.strip()]
    
    # Run tests button
    if st.button("Run Tests"):
        if not queries:
            st.error("Please enter at least one test query")
            return
            
        with st.spinner("Running tests..."):
            # Run tests asynchronously
            results = asyncio.run(run_tests(
                queries=queries,
                model_server=model_server,
                rag_model=rag_model,
                eval_model=eval_model,
                data_path=data_path,
                data_source_type=data_source_type
            ))
        
        # Display results
        st.header("Test Results")
        
        # Models used
        st.subheader("Models Configuration")
        st.write(f"RAG System Model: {model_server}/{rag_model}")
        st.write(f"Evaluation Model: {model_server}/{eval_model}")
        
        # Performance metrics
        st.subheader("Performance Metrics")
        perf_df = pd.DataFrame({
            "Metric": ["Latency (s)", "Throughput (req/s)", "Success Rate", "Error Rate"],
            "Value": [
                results["performance_metrics"].latency,
                results["performance_metrics"].throughput,
                results["performance_metrics"].success_rate,
                results["performance_metrics"].error_rate
            ]
        })
        st.dataframe(perf_df)
        
        # RAG metrics
        st.subheader("RAG Quality Metrics")
        rag_df = pd.DataFrame({
            "Metric": ["Answer Relevancy", "Faithfulness", "Contextual Relevancy"],
            "Score": [
                results["rag_metrics"].answer_relevancy,
                results["rag_metrics"].faithfulness,
                results["rag_metrics"].contextual_relevancy
            ]
        })
        st.dataframe(rag_df)
        
        # Individual test results
        st.subheader("Individual Test Results")
        for i, result in enumerate(results["results"], 1):
            with st.expander(f"Test {i}: {result['query'][:50]}..."):
                st.write("Query:", result["query"])
                if result.get("error"):
                    st.error(f"Error: {result['error']}")
                else:
                    st.write("Response:", result["response"])
                    st.write("Contexts:", result["contexts"])
                    st.write("Latency:", f"{result['latency']:.2f}s")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = Path(f"validation/results/test_results_{timestamp}.json")
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add models info to results before saving
        results["models_info"] = {
            "rag_model": rag_model,
            "eval_model": eval_model,
            "model_server": model_server
        }
        
        with open(results_path, "w") as f:
            json.dump(results, f, default=str)
        st.success(f"Results saved to {results_path}")

async def run_tests(queries, model_server, rag_model, eval_model, data_path, data_source_type):
    """Async function to run the tests"""
    async with RAGSystemTester("http://localhost:8000", metrics_collector) as tester:
        # Update data source
        if data_source_type == "Single File":
            await tester.update_data_source(filepath=data_path)
        else:
            await tester.update_data_source(data_dir=data_path)
        
        # Configure metrics collector with evaluation model
        metrics_collector.critic_llm = None  # Reset to force reinitialization with new model
        
        # Run batch test
        results = await tester.run_batch_test(
            queries=queries,
            model_server=model_server,
            model_name=rag_model  # Use RAG model for generating responses
        )
        
        # Update test results with evaluation model metrics
        if results["results"]:
            successful_results = [r for r in results["results"] if not r.get("error")]
            if successful_results:
                rag_metrics_list = []
                for result in successful_results:
                    metrics = await metrics_collector.collect_rag_metrics(
                        result["query"],
                        result["contexts"],
                        result["response"],
                        eval_model  # Use evaluation model for metrics
                    )
                    rag_metrics_list.append(metrics)
                
                # Update RAG metrics in results
                results["rag_metrics"].answer_relevancy = sum(m.answer_relevancy for m in rag_metrics_list) / len(rag_metrics_list)
                results["rag_metrics"].faithfulness = sum(m.faithfulness for m in rag_metrics_list) / len(rag_metrics_list)
                results["rag_metrics"].contextual_relevancy = sum(m.contextual_relevancy for m in rag_metrics_list) / len(rag_metrics_list)
        
        return results

if __name__ == "__main__":
    main()
