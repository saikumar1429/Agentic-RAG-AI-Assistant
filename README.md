Agentic RAG AI Assistant


This project is an Agentic Retrieval-Augmented Generation (RAG) AI Assistant that allows users to upload documents and ask questions based on their content. The system retrieves relevant information from uploaded files and generates answers grounded in the document context.

The application supports PDF, TXT, and DOCX files, processes them into embeddings, stores them in a FAISS vector database, and retrieves relevant chunks using semantic search.

An agent workflow built with LangGraph manages the process of retrieval, document grading, query transformation, and response generation.

Project implementation: 

app

Features

Document-based question answering

Supports PDF, TXT, DOCX file uploads

Semantic search using vector embeddings

Automated document relevance grading

Intelligent query rewriting if results are weak

Context-aware responses with citations

Interactive Streamlit chat interface

Agent workflow using LangGraph

Tech Stack

Python

Streamlit

LangChain

LangGraph

HuggingFace Embeddings

FAISS Vector Database

Groq LLM (Llama 3.1)

Project Architecture

The system follows an Agentic RAG pipeline:

User uploads documents

Documents are split into chunks

Embeddings are generated

Chunks are stored in a FAISS vector database

User asks a question

System retrieves relevant documents

Documents are graded for relevance

If needed, the query is rewritten

LLM generates the final answer based on context
