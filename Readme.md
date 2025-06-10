# 🤖 AI QA RAG Chatbot
A powerful AI chatbot built with LangChain, ChromaDB, and OpenAI GPT-4o, designed to answer questions from multiple uploaded documents of PDFs. Powered by Retrieval-Augmented Generation (RAG), this app allows users to interactively query document content with intelligent, context-aware responses.

## 🧠 Features

🧾 PDF + Image OCR (using pdfplumber and pytesseract)

🧠 RAG-based QA using LangChain and OpenAI models

📚 ChromaDB vector store for fast semantic search

🪄 Real-time chat interface with memory

🔍 Streamlit UI for seamless interaction

📦 Metadata sanitization to ensure vector store compatibility

## 🔧 Tech Stack
Component	Description
🧠 LLM	OpenAI GPT-4o (ChatOpenAI)
📚 Embeddings	OpenAIEmbeddings
🗂 Vector Store	ChromaDB with persistent client
🧱 Framework	LangChain (Expression Language & RAG)
🖼 OCR	Tesseract (pytesseract)
🧾 PDF Parser	pdfplumber
🎛 Frontend	Streamlit

## 🚀 How It Works

Documents are:

Parsed and split into smaller chunks.

Cleaned and embedded into a Chroma vector store.

A retriever fetches relevant chunks based on your question.

The chatbot generates an accurate answer using context-aware prompting and GPT-4o.

You chat in real-time with AI about your uploaded content!
