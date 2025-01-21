import time
import functools
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric
)
from deepeval.test_case import LLMTestCase
import logging
from langfuse import Langfuse
from utils.ollama_api import OllamaEvalLLM

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def retry(retry_count=3, first_delay=0, backoff=2, retry_exceptions=(TimeoutError,)):
    def _retry(func):
        @functools.wraps(func)
        def inner(*args, **kwargs):
            tries, delay = retry_count, first_delay
            while tries > 1:
                try:
                    return func(*args, **kwargs)
                except Exception as exc:  # noqa: W0703
                    if not any(isinstance(exc, r_exc) for r_exc in retry_exceptions):
                        print(f'Got error {exc}. It is not "retry exception"!')
                        raise
                    print(f'Got error {exc}, retry in {delay} sec...')
                    time.sleep(delay)
                    tries -= 1
                    delay *= backoff
            return func(*args, **kwargs)
        return inner
    return _retry

@dataclass
class PerformanceMetrics:
    latency: float
    throughput: float
    success_rate: float
    error_rate: float

@dataclass
class RAGMetrics:
    answer_relevancy: float
    faithfulness: float
    contextual_relevancy: float

class ValidationMetricsCollector:
    def __init__(self, langfuse_client: Langfuse):
        self.langfuse = langfuse_client
        self._init_metrics()
        
    def _init_metrics(self):
        """Initialize DeepEval metrics with critic LLM"""
        try:
            self.critic_llm = OllamaEvalLLM("qwen2.5:7b-instruct-q4_0")
            
            # Initialize metrics with custom critic LLM
            self.metrics = {
                'answer_relevancy': AnswerRelevancyMetric(
                    threshold=0.5,
                    verbose_mode=True,
                    model=self.critic_llm,
                    include_reason=True
                ),
                'faithfulness': FaithfulnessMetric(
                    threshold=0.5,
                    verbose_mode=True,
                    model=self.critic_llm,
                    include_reason=True,
                    truths_extraction_limit=50
                ),
                'contextual_relevancy': ContextualRelevancyMetric(
                    threshold=0.5,
                    verbose_mode=True,
                    model=self.critic_llm,
                    include_reason=True
                )
            }
                    
        except Exception as e:
            logger.error(f"Error initializing DeepEval metrics: {str(e)}")
            raise

    def _sanitize_input(self, text: str) -> str:
        """Sanitize input text to prevent parsing errors"""
        if not isinstance(text, str):
            return str(text)
        return text.strip()

    def _validate_contexts(self, contexts: List[str]) -> List[str]:
        """Validate and clean context entries"""
        if not contexts:
            return []
        return [self._sanitize_input(ctx) for ctx in contexts if ctx]

    async def collect_performance_metrics(self, test_results: List[Dict[str, Any]]) -> PerformanceMetrics:
        """Calculate performance metrics with retry logic"""
        try:
            total_time = sum(r.get('latency', 0) for r in test_results)
            total_requests = len(test_results)
            errors = sum(1 for r in test_results if r.get('error'))
            
            return PerformanceMetrics(
                latency=total_time / total_requests if total_requests > 0 else 0,
                throughput=total_requests / total_time if total_time > 0 else 0,
                success_rate=(total_requests - errors) / total_requests if total_requests > 0 else 0,
                error_rate=errors / total_requests if total_requests > 0 else 0
            )
        except Exception as e:
            logger.error(f"Error collecting performance metrics: {str(e)}")
            raise

    @retry(retry_count=3)
    async def collect_rag_metrics(self, 
                                query: str,
                                contexts: List[str],
                                response: str) -> Optional[RAGMetrics]:
        """Calculate RAG metrics using DeepEval"""
        # Sanitize inputs
        clean_query = self._sanitize_input(query)
        clean_contexts = self._validate_contexts(contexts)
        clean_response = self._sanitize_input(response)
        
        # Log inputs for debugging
        logger.info(f"Processing RAG metrics for query: {clean_query[:100]}...")
        logger.info(f"Number of contexts: {len(clean_contexts)}")
        
        # Create test case
        test_case = LLMTestCase(
            input=clean_query,
            actual_output=clean_response,
            retrieval_context=clean_contexts
        )
        
        scores = {}
        for metric_name, metric in self.metrics.items():
            metric.measure(test_case)
            scores[metric_name] = float(metric.score)
        
        return RAGMetrics(
            answer_relevancy=scores.get('answer_relevancy', 0),
            faithfulness=scores.get('faithfulness', 0),
            contextual_relevancy=scores.get('contextual_relevancy', 0)
        )

    async def log_metrics_to_langfuse(self,
                                    test_id: str,
                                    performance_metrics: PerformanceMetrics,
                                    rag_metrics: Optional[RAGMetrics]) -> bool:
        """Log metrics to Langfuse with error handling"""
        try:
            trace = self.langfuse.trace(
                id=test_id,
                name="rag_validation"
            )
            
            # Log performance metrics
            if performance_metrics:
                for metric_name, value in performance_metrics.__dict__.items():
                    trace.score(name=metric_name, value=float(value))
            
            # Log RAG metrics if available
            if rag_metrics:
                for metric_name, value in rag_metrics.__dict__.items():
                    trace.score(name=metric_name, value=float(value))
            
            return True
            
        except Exception as e:
            logger.error(f"Error logging to Langfuse: {str(e)}")
            return False
