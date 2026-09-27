# RAG Restaurant

A Python-based Retrieval-Augmented Generation (RAG) application for answering questions about restaurant reviews using a local vector database and a LangChain + Ollama pipeline.

## Overview

This project loads restaurant review data from a CSV file, creates embeddings with Ollama, stores them in a Chroma vector database, and answers natural-language questions using a retrieval-augmented generation workflow.

The application exposes a FastAPI service that accepts questions and returns an answer along with the supporting review snippets used to generate the response.

## Features

- FastAPI API for query handling
- Restaurant review dataset ingestion from CSV
- Embedding generation with Ollama
- Vector search using Chroma
- Retrieval-augmented answer generation with LangChain
- Health-check and query endpoints

## Project Structure

```text
.
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── rag_pipeline.py
│   └── vector_store.py
├── chrome_langchain_db/
├── data/
│   └── realistic_restaurant_reviews.csv
├── requirements.txt
├── README.md
└── .gitignore