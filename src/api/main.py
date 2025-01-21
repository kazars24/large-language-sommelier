from fastapi import FastAPI, Depends

from src.api.services import get_recommendation_service, RecommendationService
from src.core.models import Query, ModelChoice
from src.core.config import settings
import os

app = FastAPI(
    title="Wine Recommendation API",
    description="API для получения рекомендаций по винам",
    version="1.2.0"
)

@app.get("/")
async def root():
    return {"message": "Wine Recommendation API is running"}

@app.post("/recommend/")
async def get_recommendation(query: Query, model_choice: ModelChoice, rag_service: RecommendationService = Depends(get_recommendation_service)):
    print(f"Received request at /recommend/ with:")
    print(f"  Query: {query}")
    print(f"  Model Choice: {model_choice}")
    
    # Get both response and contexts
    res = rag_service.get_recommendation_with_context(
        query.question,
        model_choice.model_server,
        model_choice.model_name
    )
    response = res['recommendation']
    contexts = res['retrieved_contexts']['context']
    
    return {"response": response, "question": query.question, "contexts": contexts}

@app.post("/update_data/")
async def update_rag_data(filepath: str = None, data_dir: str = None):
    """
    Update the data used by the RAG system.
    Handles both single file uploads and directory paths for multiple files.
    """
    global rag_service

    if filepath:
        # Process a single file
        settings.DATA_FILEPATH = filepath
    elif data_dir:
        # Process a directory
        settings.RAG_DATA_DIR = data_dir
    else:
        # Use default values from settings if neither filepath nor data_dir is provided
        filepath = settings.DATA_FILEPATH
        data_dir = settings.RAG_DATA_DIR

    rag_service = get_recommendation_service()

    if filepath:
        source = f"filepath: {filepath}"
    elif data_dir:
        source = f"data directory: {data_dir}"
    else:
        source = "default data from settings"

    return {"message": "RAG data updated successfully", "source": source}

rag_service = get_recommendation_service()
