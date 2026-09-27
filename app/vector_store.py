import os
import pandas as pd
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

EMBED_MODEL = "mxbai-embed-large"
DB_LOCATION = "./chrome_langchain_db"
COLLECTION = "restaurant_reviews"
CSV_PATH = "./data/realistic_restaurant_reviews.csv"


def build_vector_store() -> Chroma:
    """Build or load the Chroma vector store."""
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    vector_store = Chroma(
        collection_name=COLLECTION,
        persist_directory=DB_LOCATION,
        embedding_function=embeddings,
    )

    # Only index if the DB folder doesn't exist yet
    if not os.path.exists(DB_LOCATION):
        print(f"🔨 Indexing reviews from {CSV_PATH} ...")
        df = pd.read_csv(CSV_PATH)

        documents = []
        ids = []
        for i, row in df.iterrows():
            documents.append(
                Document(
                    page_content=f"{row['Title']} {row['Review']}",
                    metadata={"rating": row["Rating"], "date": row["Date"]},
                )
            )
            ids.append(str(i))

        vector_store.add_documents(documents=documents, ids=ids)
        print(f"✅ Indexed {len(documents)} reviews.")
    else:
        print("📦 Loaded existing vector store.")

    return vector_store


def get_retriever():
    """Return a retriever configured for top-5 similarity search."""
    vector_store = build_vector_store()
    return vector_store.as_retriever(search_kwargs={"k": 5})