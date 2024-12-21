from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from models import SummarizeRequest
from langfuse_setup import setup_langfuse
from ollama_api import OllamaAPIWrapper
from ragas_evaluation import SummaryEvaluator
from config import OLLAMA_API_BASE

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

langfuse = setup_langfuse()
ollama_wrapper = OllamaAPIWrapper(base_url=OLLAMA_API_BASE)
summary_evaluator = SummaryEvaluator(ollama_llm=ollama_wrapper.llm)

@app.get("/models")
async def get_models():
    """Returns a list of available Ollama models."""
    return await ollama_wrapper.get_models()

@app.post("/summarize")
async def summarize(request: SummarizeRequest):
    """Generates and evaluates a summary using Ollama."""

    trace = langfuse.trace(name=f"summary-generation-{request.model}")

    ollama_wrapper.update_llm(model=request.model)
    summary_evaluator = SummaryEvaluator(ollama_llm=ollama_wrapper.llm)

    try:
        span = trace.span(name="generate_summary", input = {'text': request.text})
        summary, processing_time, input_tokens, output_tokens, throughput = ollama_wrapper.generate_summary(
            text=request.text
        )
        span.end(
            output=summary
        )

        span = trace.span(name="evaluate_summary", input = {'text': request.text})
        ragas_scores = summary_evaluator.evaluate_summary(
            text=request.text,
            summary=summary
        )
        faithfulness_score = float(ragas_scores["faithfulness"][0])
        relevancy_score = float(ragas_scores["answer_relevancy"][0])
        eval_metrics = {
            "throughput": throughput,
            "faithfulness_score": faithfulness_score,
            "relevancy_score": relevancy_score,
        }
        span.end(
            output=eval_metrics
        )


        trace.score(
            name="faithfulness",
            value=faithfulness_score,
            comment=f"Faithfulness score for {request.model}"
        )
        trace.score(
            name="answer_relevancy",
            value=relevancy_score,
            comment=f"Answer relevancy score for {request.model}"
        )
        trace.score(
            name="throughput",
            value=throughput,
            comment=f"Throughput (tokens/s) for {request.model}"
        )

        return {
            "summary": summary,
            "metrics": eval_metrics
        }

    except Exception as e:
        raise e
