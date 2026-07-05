import time


def print_header(title):
    print("=" * 40)
    print(title)
    print("=" * 40)


def timer_start():
    return time.time()


def timer_end(start_time):
    return round(time.time() - start_time, 2)


def format_score(score):
    return round(float(score), 4)


def build_metrics_report(embedding_dim, chunk_size, chunk_overlap, top_k, avg_time):
    report = {
        "Embedding Model": "sentence-transformers/all-MiniLM-L6-v2",
        "Embedding Dimension": embedding_dim,
        "Chunk Size": chunk_size,
        "Chunk Overlap": chunk_overlap,
        "Vector Database": "FAISS",
        "Retriever": "DocumentRetriever (Similarity Search)",
        "LLM": "Groq - llama-3.3-70b-versatile",
        "Top-K": top_k,
        "Average Retrieval Time (sec)": avg_time
    }
    return report
