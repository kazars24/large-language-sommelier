from fastapi import FastAPI, Depends

from src.api.services import get_recommendation_service, RecommendationService
from src.core.models import Query, ModelChoice
from src.core.config import settings


app = FastAPI(
    title="Wine Recommendation API",
    description="API для получения рекомендаций по винам",
    version="1.1.0"
)


@app.get("/")
async def root():
    return {"message": "Wine Recommendation API is running"}


@app.post("/recommend/")
async def get_recommendation(query: Query, model_choice: ModelChoice, rag_service: RecommendationService = Depends(get_recommendation_service)):
    print(f"Received request at /recommend/ with:")
    print(f"  Query: {query}")
    print(f"  Model Choice: {model_choice}")

    response = rag_service.get_recommendation(query.question, model_choice.model_server, model_choice.model_name)
    return {"response": response}


@app.post("/update_data/")
async def update_rag_data(filepath: str = None, source_column: str = None):
    """
    Update the data used by the RAG system.
    If filepath and source_column are provided, use them.
    Otherwise, use the default values from settings.
    """
    global rag_service  # Access the global variable

    if filepath and source_column:
        settings.DATA_FILEPATH = filepath
        settings.DATA_SOURCE_COLUMN = source_column
    else:
        filepath = settings.DATA_FILEPATH
        source_column = settings.DATA_SOURCE_COLUMN

    # Reinitialize the RAG system with the new data
    rag_service = get_recommendation_service()

    return {"message": "RAG data updated successfully", "filepath": filepath, "source_column": source_column}

# Initialize rag_service here to make it available globally
rag_service = get_recommendation_service()
