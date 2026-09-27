from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

from app.vector_store import get_retriever

LLM_MODEL = "llama3.2:3b"

TEMPLATE = """
You are an expert in answering questions about a pizza restaurant.

Here are some relevant reviews: {reviews}

Here is the question to answer: {question}
"""


def get_rag_chain():
    """Build the LangChain chain: prompt | LLM."""
    model = OllamaLLM(model=LLM_MODEL)
    prompt = ChatPromptTemplate.from_template(TEMPLATE)
    return prompt | model


def answer_question(chain, retriever, question: str):
    """Run retrieval + generation for a single question."""
    reviews = retriever.invoke(question)
    answer = chain.invoke({"reviews": reviews, "question": question})
    return answer, reviews