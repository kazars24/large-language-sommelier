from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
from dotenv import load_dotenv
from langchain_gigachat.chat_models import GigaChat
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate

# Инициализация FastAPI
app = FastAPI(
    title="Wine Recommendation API",
    description="API для получения рекомендаций по винам",
    version="1.0.0"
)

# Модель для запроса
class Query(BaseModel):
    question: str

# Загрузка и инициализация компонентов RAG-системы
def initialize_rag_system():
    load_dotenv()
    
    giga_key = os.environ.get("GIGACHAT_CREDENTIALS")
    giga = GigaChat(credentials=giga_key, model="GigaChat", timeout=30, verify_ssl_certs=False)
    giga.verbose = False
    print('Model is initialized')

    loader = CSVLoader('data/wine_desc.csv', source_column='Название')
    data = loader.load()
    print('Data is loaded')

    embeddings_model = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-small",
        multi_process=False,
        encode_kwargs={"normalize_embeddings": True})

    print('embeddings_model is initialized')

    db = Chroma.from_documents(data, embeddings_model)
    print('Chroma is initialized')

    template = (
        "Ты являешься профессиональным сомелье и помощником по подбору вин. "
        "Используй приведенные ниже фрагменты из извлеченного контекста, чтобы ответить на вопрос и помочь пользователю. "
        "Если ты не знаешь ответа, просто скажи, что ты не знаешь. "
        "Предлагай максимум три варианта.\n"
        "Вопрос: {question}\n"
        "Контекст: {context}\n"
        "Ответ:"
    )

    prompt = PromptTemplate(template=template, input_variables=['context', 'question'])
    retriever = db.as_retriever()

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | giga
        | StrOutputParser()
    )
    print('rag_chain is initialized')

    return rag_chain

# Инициализация RAG-системы при запуске
rag_chain = initialize_rag_system()

@app.get("/")
async def root():
    return {"message": "Wine Recommendation API is running"}

@app.post("/recommend/")
async def get_recommendation(query: Query):
    try:
        # Получение ответа от RAG-системы
        response = rag_chain.invoke(query.question)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Запуск сервера
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)