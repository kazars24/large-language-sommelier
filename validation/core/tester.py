import time
import httpx
import asyncio
from typing import List, Dict, Any
from datetime import datetime
import uuid
import sys, os
from core.metrics import RAGMetrics

import platform
if platform.system()=='Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class RAGSystemTester:
    def __init__(self, base_url: str, metrics_collector, ollama_url: str = "http://192.168.1.198:11434"):
        self.base_url = base_url
        self.ollama_url = ollama_url
        self.metrics_collector = metrics_collector
        self.client = None
    
    async def __aenter__(self):
        """Setup async context manager"""
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup async context manager"""
        if self.client:
            await self.client.aclose()

    async def get_models(self) -> List[str]:
        """Retrieves available models from the Ollama API."""
        try:
            response = await self.client.get(f"{self.ollama_url}/api/tags")
            response.raise_for_status()
            models = response.json().get("models", [])
            return [model["name"] for model in models]
        except Exception as e:
            print(f"Error fetching models: {e}")
            return ["qwen2.5:7b-instruct-q4_0"]  # Default fallback
    
    async def update_data_source(self, filepath: str = None, data_dir: str = None) -> bool:
        """Update RAG system data source"""
        if not self.client:
            raise RuntimeError("Client not initialized. Use async context manager.")
            
        params = {}
        if filepath:
            params['filepath'] = filepath
        if data_dir:
            params['data_dir'] = data_dir
            
        response = await self.client.post(
            f"{self.base_url}/update_data/",
            params=params,
            timeout=None
        )
        return response.status_code == 200
    
    async def test_single_query(self, 
                              query: str,
                              model_server: str = "ollama",
                              model_name: str = "qwen2.5:7b-instruct-q4_0") -> Dict[str, Any]:
        """Test single query and collect metrics"""
        if not self.client:
            raise RuntimeError("Client not initialized. Use async context manager.")
            
        start_time = time.time()
        request_data = {
            "query": {"question": query},
            "model_choice": {
                "model_server": model_server,
                "model_name": model_name
            }
        }
        
        response = await self.client.post(
            f"{self.base_url}/recommend/",
            json=request_data,
            timeout=None
        )

        result = response.json()
        latency = time.time() - start_time

        trace = self.metrics_collector.langfuse.trace(
            name=f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )

        trace.span(
            name="input",
            input={"query": query, "model": f"{model_server}/{model_name}"}
        )

        trace.span(
            name="output",
            output={
                "response": result["response"],
                "contexts": result.get("contexts", []),
                "latency": latency
            }
        )
        
        return {
            "query": query,
            "response": result["response"],
            "contexts": result.get("contexts", []),
            "latency": latency,
            "error": None,
            "trace_id": trace.id
        }
    
    async def run_batch_test(self,
                            queries: List[str],
                            model_server: str = "ollama",
                            model_name: str = "qwen2.5:7b-instruct-q4_0") -> Dict[str, Any]:
        """Run batch test with multiple queries"""
        if not self.client:
            raise RuntimeError("Client not initialized. Use async context manager.")
            
        test_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Run tests in parallel
        tasks = [
            self.test_single_query(query, model_server, model_name)
            for query in queries
        ]
        test_results = await asyncio.gather(*tasks)
        
        # Collect metrics
        performance_metrics = await self.metrics_collector.collect_performance_metrics(test_results)
        
        # Calculate RAG metrics for successful queries
        rag_metrics_list = []
        for result in test_results:
            if not result.get("error"):
                rag_metrics = await self.metrics_collector.collect_rag_metrics(
                    result["query"],
                    result["contexts"],
                    result["response"],
                    model_name
                )
                rag_metrics_list.append(rag_metrics)
        
        # Average RAG metrics
        avg_rag_metrics = RAGMetrics(
            answer_relevancy=sum(m.answer_relevancy for m in rag_metrics_list) / len(rag_metrics_list),
            faithfulness=sum(m.faithfulness for m in rag_metrics_list) / len(rag_metrics_list),
            contextual_relevancy=sum(m.contextual_relevancy for m in rag_metrics_list) / len(rag_metrics_list)
        )
        
        # Log metrics to Langfuse
        await self.metrics_collector.log_metrics_to_langfuse(
            test_id,
            performance_metrics,
            avg_rag_metrics
        )
        
        return {
            "test_id": test_id,
            "timestamp": timestamp,
            "model_server": model_server,
            "model_name": model_name,
            "performance_metrics": performance_metrics,
            "rag_metrics": avg_rag_metrics,
            "results": test_results
        }
