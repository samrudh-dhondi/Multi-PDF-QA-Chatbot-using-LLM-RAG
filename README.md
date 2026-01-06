# Multi-PDF-QA-Chatbot-using-LLM-RAG
A system designed to extract and search information from multiple PDF files using vector embeddings to retrieve relevant document context before generating accurate responses. The project demonstrates an end-to-end RAG pipeline with text extraction, chunking, vector storage, and LLM-based response generation using open-source tools.

This project is a **Streamlit-based web application** that enables users to upload multiple PDF documents and ask natural language questions about their content.  
It uses a **Retrieval-Augmented Generation (RAG)** approach to fetch relevant information from documents and generate accurate answers using **open-source Large Language Models (LLMs)** hosted on Hugging Face.

The application is designed to work efficiently with free-tier models and provides a clean chat-style interface for interaction.

---

## 🔍 What This Project Does

- Accepts **multiple PDF documents** from the user
- Extracts and processes text from PDFs
- Splits large documents into meaningful chunks
- Converts text chunks into vector embeddings
- Stores embeddings in a **FAISS vector database**
- Retrieves relevant document context based on user queries
- Generates answers using **Hugging Face LLMs**
- Maintains short chat history for better conversational responses

---

## 🧠 Core Concepts Used

- **Retrieval-Augmented Generation (RAG)**
- **Vector Similarity Search**
- **Semantic Text Embeddings**
- **Context-aware LLM prompting**
- **Fallback-based model handling**

---

## 📁 Project Files Overview

### `app.py`
Main application file that:
- Handles PDF upload and processing
- Extracts text using PyPDF2
- Splits text into overlapping chunks
- Generates embeddings using Sentence Transformers
- Stores and queries vectors using FAISS
- Connects to Hugging Face Inference API
- Manages chat history and user interaction
- Renders the Streamlit UI

---

### `htmlTemplates.py`
Contains UI-related components:
- Custom CSS for chat layout
- HTML templates for user messages
- HTML templates for bot responses
- Avatar-based chat message formatting

---

## 🤖 Models & AI Components

### Embedding Model
- `sentence-transformers/all-MiniLM-L6-v2`

### Language Models (Fallback-based)
The system automatically tries multiple models for better reliability:
- Mistral 7B Instruct
- Phi-3 Mini
- Zephyr 7B
- LLaMA 2 (Chat)
- BART (used as summarization fallback)

---

## 🎨 User Interface

- Chat-style UI with user and bot message separation
- Sidebar for document upload and processing
- Visual feedback for loading, errors, and processing status
- Custom-styled messages using HTML & CSS

---

## 🎯 Use Case Examples

- Document Q&A systems
- Research paper exploration
- Resume or policy document analysis
- Academic or corporate PDF assistants
- Knowledge extraction from multiple documents

---

## ⭐ Final Note

This project demonstrates a **practical, real-world implementation of RAG** using:
- Streamlit
- LangChain
- Hugging Face
- FAISS
- Open-source LLMs

It is suitable for portfolios, learning projects, and further production-level extensions.
