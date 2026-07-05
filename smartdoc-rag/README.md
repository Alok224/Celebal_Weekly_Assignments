# SmartDoc - RAG Based Document Question Answering

## Project Overview

SmartDoc is a Retrieval-Augmented Generation (RAG) application that lets you upload your own documents (PDF or TXT) and ask questions about them in plain English. Instead of relying purely on an LLM's internal knowledge, the system retrieves the most relevant chunks of your document using a FAISS vector store and then asks a Groq-hosted LLM (Llama 3.3 70B) to answer strictly from that retrieved context.

This project was built as a hands-on Generative AI exercise to understand how retrieval and generation work together in a real application.

## Features

- Upload PDF or TXT documents through a simple Streamlit UI
- Automatic text cleaning and chunking with configurable chunk size / overlap
- Local embedding generation using `sentence-transformers/all-MiniLM-L6-v2`
- FAISS vector index with automatic save/reload
- Simple similarity-based `DocumentRetriever` with visible similarity scores
- Grounded answer generation using Groq's Llama-3.3-70B-Versatile
- Built-in evaluation section that runs 5 sample questions and shows retrieved context + answers
- Metrics report (embedding model, chunk size, retriever, LLM, average retrieval time, etc.)

## Architecture

The pipeline follows the standard RAG flow:

1. **Document Ingestion** - PDF/TXT files are loaded and converted to raw text
2. **Text Cleaning** - extra whitespace removed, empty chunks dropped
3. **Chunking** - `RecursiveCharacterTextSplitter` splits text into overlapping chunks
4. **Embedding** - each chunk is embedded using MiniLM sentence-transformers
5. **Vector Store** - embeddings are stored in a local FAISS index
6. **Retrieval** - user query is embedded and top-K similar chunks are fetched
7. **Generation** - retrieved chunks + question are sent to Groq LLM to produce a grounded answer

## Folder Structure

```
smartdoc-rag/
│
├── data/                  # uploaded documents get stored here
│
├── src/
│   ├── app.py             # Streamlit UI
│   ├── chatbot.py         # prompt building + Groq answer generation
│   ├── document_loader.py # loading, cleaning, chunking
│   ├── retriever.py       # DocumentRetriever (FAISS similarity search)
│   ├── vectorstore.py     # FAISS build / save / load
│   └── utils.py           # small helper functions
│
├── requirements.txt
├── README.md
└── .env.example
```

## Installation

1. Clone this repository and move into the folder

```bash
git clone <your-repo-url>
cd smartdoc-rag
```

2. Create a virtual environment (Python 3.13 recommended)

```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Set up your environment variables

```bash
cp .env.example .env
```

Add your Groq API key inside `.env`:

```
GROQ_API_KEY=your_groq_api_key_here
```

You can get a free API key from [console.groq.com](https://console.groq.com).

## Usage

Run the Streamlit app from the project root:

```bash
streamlit run src/app.py
```

Then in the browser:

1. Upload a PDF or TXT file
2. Click **Build Vector Database**
3. Type a question and click **Get Answer**
4. Optionally click **Run Sample Evaluation** to see the system answer 5 auto-generated questions

## Workflow

```
Upload Document -> Clean & Chunk Text -> Generate Embeddings -> Build FAISS Index
        -> Ask Question -> Retrieve Top-K Chunks -> Build Prompt -> Groq LLM -> Answer
```

## Screenshots

_Add screenshots of the Streamlit UI here after running the app locally._

## Future Improvements

- Add hybrid search (keyword + vector) for better retrieval
- Add a re-ranking step on top of FAISS results
- Support more file types (DOCX, CSV)
- Add chat history / multi-turn conversation memory
- Deploy on Streamlit Cloud or HuggingFace Spaces

## Tech Stack

- Python 3.13
- Streamlit
- LangChain / LangChain-Groq
- HuggingFace Embeddings (`all-MiniLM-L6-v2`)
- FAISS
- PyMuPDF, pypdf
- python-dotenv
