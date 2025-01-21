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

# Initialize Langfuse client
langfuse = Langfuse(
    public_key='pk-lf-7d30e5df-41a6-4888-9602-2e5e19e6ef73',
    secret_key='sk-lf-2195fe3e-eb64-4bdc-8891-072f603e528f',
    host='http://localhost:3000'
)

# Initialize metrics collector
metrics_collector = ValidationMetricsCollector(langfuse)

def load_test_queries():
    """Load sample test queries"""
    return [
        "Порекомендуй красное вино к ужину",
        "Какое вино подойдет к рыбе?",
        "Порекомендуй сладкое десертное вино",
        "Вино с нотками персика"
    ]

async def run_tests(queries, model_server, model_name, data_path, data_source_type):
    """Async function to run the tests"""
    async with RAGSystemTester("http://localhost:8000", metrics_collector) as tester:
        # Update data source
        if data_source_type == "Single File":
            await tester.update_data_source(filepath=data_path)
        else:
            await tester.update_data_source(data_dir=data_path)
        
        # Run batch test
        results = await tester.run_batch_test(
            queries=queries,
            model_server=model_server,
            model_name=model_name
        )
        return results

def main():
    st.title("Wine Recommendation RAG System Validator")
    
    # Sidebar configuration
    st.sidebar.header("Test Configuration")
    
    # Model selection
    model_server = st.sidebar.selectbox(
        "Model Server",
        ["ollama", "gigachat"],
        index=0
    )
    
    if model_server == "ollama":
        model_name = st.sidebar.selectbox(
            "Model Name",
            ["qwen2.5:7b-instruct-q4_0"],
            index=0
        )
    else:
        model_name = st.sidebar.selectbox(
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
        with st.spinner("Running tests..."):
            # Run tests asynchronously
            results = asyncio.run(run_tests(
                queries=queries,
                model_server=model_server,
                model_name=model_name,
                data_path=data_path,
                data_source_type=data_source_type
            ))
        
        # Display results
        st.header("Test Results")
        
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
            "Metric": ["answer_relevancy", "faithfulness", "contextual_relevancy"],
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
        with open(results_path, "w") as f:
            json.dump(results, f, default=str)
        st.success(f"Results saved to {results_path}")

if __name__ == "__main__":
    main()
