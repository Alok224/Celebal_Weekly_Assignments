import sys
import os
sys.path.append(os.path.dirname(__file__))

import streamlit as st

from document_loader import load_document, chunk_text
from vectorstore import build_vectorstore, get_embedding_model
from retriever import DocumentRetriever
from chatbot import generate_answer, run_evaluation
from utils import build_metrics_report

st.set_page_config(page_title="SmartDoc RAG", layout="wide")
st.title("SmartDoc - Document Question Answering (RAG)")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

st.sidebar.header("Settings")
chunk_size = st.sidebar.slider("Chunk Size", 200, 1500, 500, 50)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 300, 50, 10)
top_k = st.sidebar.slider("Top K Chunks", 1, 10, 3)

st.header("1. Upload Document")
uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

if uploaded_file is not None:
    os.makedirs("data", exist_ok=True)
    file_path = os.path.join("data", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("File saved: " + uploaded_file.name)

    if st.button("Build Vector Database"):
        with st.spinner("Processing document..."):
            pages = load_document(file_path)
            chunks = chunk_text(pages, chunk_size, chunk_overlap)
            db = build_vectorstore(chunks)
            st.session_state.vectorstore = db
            st.session_state.chunk_count = len(chunks)
        st.success("Vector database built with " + str(len(chunks)) + " chunks")

st.header("2. Ask a Question")
question = st.text_input("Enter your question")

if st.button("Get Answer"):
    if st.session_state.vectorstore is None:
        st.warning("Please build the vector database first.")
    elif question.strip() == "":
        st.warning("Please enter a question.")
    else:
        retriever = DocumentRetriever(st.session_state.vectorstore, top_k=top_k)
        chunks, scores = retriever.retrieve(question)
        answer, inference_time = generate_answer(chunks, question)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Retrieved Chunks")
        for i, (c, s) in enumerate(zip(chunks, scores)):
            st.markdown(f"**Chunk {i + 1} | Similarity Score: {round(float(s), 4)}**")
            st.write(c)

        st.subheader("Metrics")
        st.write("Number of Chunks Retrieved:", len(chunks))
        st.write("Inference Time (sec):", inference_time)

st.header("3. Evaluation")
if st.button("Run Sample Evaluation"):
    if st.session_state.vectorstore is None:
        st.warning("Please build the vector database first.")
    else:
        sample_questions = [
            "What is this document about?",
            "Summarize the main points.",
            "What are the key findings?",
            "List any important dates mentioned.",
            "What conclusions are drawn in the document?"
        ]
        retriever = DocumentRetriever(st.session_state.vectorstore, top_k=top_k)
        results = run_evaluation(retriever, sample_questions)

        avg_time = round(sum(r["time"] for r in results) / len(results), 2)

        for r in results:
            st.markdown(f"**Q: {r['question']}**")
            st.write("Answer:", r["answer"])
            with st.expander("Retrieved Context"):
                for c in r["context"]:
                    st.write(c)
            st.write("---")

        st.subheader("Metrics Report")
        embeddings = get_embedding_model()
        embedding_dim = len(embeddings.embed_query("test"))
        report = build_metrics_report(embedding_dim, chunk_size, chunk_overlap, top_k, avg_time)
        st.json(report)
