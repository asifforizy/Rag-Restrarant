from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.rag_pipeline import get_rag_chain, answer_question
from app.vector_store import get_retriever

app = FastAPI(title="Pizza Restaurant Review RAG API")

# Global state (populated at startup)
retriever = None
chain = None


@app.on_event("startup")
def startup():
    global retriever, chain
    print("🚀 Starting up RAG pipeline...")
    retriever = get_retriever()
    chain = get_rag_chain()
    print("✅ RAG pipeline ready.")


class QueryRequest(BaseModel):
    question: str


class SourceDoc(BaseModel):
    rating: int | str | None = None
    date: str | None = None
    content: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceDoc]


@app.get("/")
def root():
    return {"message": "Pizza RAG API is running. POST to /query."}


@app.get("/health")
def health():
    return {"status": "ok", "ready": chain is not None}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if chain is None or retriever is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not ready yet.")

    answer, reviews = answer_question(chain, retriever, request.question)

    sources = [
        SourceDoc(
            rating=doc.metadata.get("rating"),
            date=str(doc.metadata.get("date")),
            content=doc.page_content,
        )
        for doc in reviews
    ]

    return QueryResponse(question=request.question, answer=answer, sources=sources)