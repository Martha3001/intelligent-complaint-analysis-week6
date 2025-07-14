# AI-Powered Complaint Analysis Tool for CreditTrust Financial

## Overview
This project develops a Retrieval-Augmented Generation (RAG) chatbot to analyze unstructured customer complaints for CreditTrust Financial. The tool empowers internal teams (Product, Support, Compliance) to quickly identify trends and resolve issues by answering natural language questions (e.g., "What are the common issues with personal loans?").

## Key Features
- **Semantic Search**: FAISS vector database with `all-MiniLM-L6-v2` embeddings for complaint retrieval
- **LLM Synthesis**: Llama2 generates concise answers from retrieved complaints
- **Interactive UI**: Gradio web interface for non-technical users
- **Multi-Product Support**: Analyzes complaints across 5 financial products
- **Evidence-Based**: Displays source complaint excerpts for transparency

## Setup
1. Clone: `git clone https://github.com/Martha3001/intelligent-complaint-analysis-week6.git`
2. Create venv: `python -m venv .venv`
3. Activate: `.venv\Scripts\activate` (Windows)
4. Install: `pip install -r requirements.txt`

## Usage
Launch the Gradio interface:

```bash
python gradio\app.py
```

Access the UI at http://localhost:7860

Ask questions like:
    "What fraud patterns exist in money transfers?"

## Project Structure
```
intelligent-complaint-analysis-week6/
├── data/                   # Complaint datasets
├── gradio/ 
│   └── app.py              # Gradio interface
├── notebooks/             
│   ├── chunking.ipynb
│   ├── eda_and_data_processing.ipynb # EDA 
│   ├── rag.ipynb    
│   └── vector_db.ipynb
├── src/    
│   └── rag.py    # Core retrieval/generation logic
├── vector_store/           # FAISS/ChromaDB indexes 
├── .gitignore
├── README.md                
└── requirements.txt
```
