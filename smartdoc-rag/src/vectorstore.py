import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = "faiss_index"

_embedding_model = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    return _embedding_model


def build_vectorstore(chunks, index_path=INDEX_PATH):
    embeddings = get_embedding_model()
    vector_dim = len(embeddings.embed_query("test"))
    print("Embedding Dimension:", vector_dim)

    db = FAISS.from_texts(chunks, embeddings)
    db.save_local(index_path)
    print("FAISS Index Created")
    return db


def load_vectorstore(index_path=INDEX_PATH):
    if not os.path.exists(index_path):
        return None
    embeddings = get_embedding_model()
    db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    print("FAISS Index Loaded from disk")
    return db


def get_or_create_vectorstore(chunks, index_path=INDEX_PATH):
    db = load_vectorstore(index_path)
    if db is None:
        db = build_vectorstore(chunks, index_path)
    return db
