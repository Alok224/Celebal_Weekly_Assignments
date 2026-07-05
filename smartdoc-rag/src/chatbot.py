import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

GROQ_MODEL = "llama-3.3-70b-versatile"
NO_ANSWER_MSG = "I could not find enough information inside the uploaded documents."

_llm = None


def get_llm():
    global _llm
    if _llm is None:
        api_key = os.getenv("GROQ_API_KEY")
        _llm = ChatGroq(model=GROQ_MODEL, groq_api_key=api_key, temperature=0)
    return _llm


def build_prompt(context, question):
    prompt = f"""Answer the question using ONLY the context given below.
If the context does not have enough information, reply exactly with:
"{NO_ANSWER_MSG}"

Context:
{context}

Question:
{question}

Answer:"""
    return prompt


def generate_answer(context_chunks, question):
    context = "\n\n".join(context_chunks)
    prompt = build_prompt(context, question)
    llm = get_llm()

    start_time = time.time()
    response = llm.invoke(prompt)
    end_time = time.time()

    inference_time = round(end_time - start_time, 2)
    print("Inference Time:", inference_time, "sec")
    print("Response Generated")

    return response.content, inference_time


def run_evaluation(retriever, questions):
    results = []
    for q in questions:
        chunks, scores = retriever.retrieve(q)
        answer, inference_time = generate_answer(chunks, q)
        results.append({
            "question": q,
            "context": chunks,
            "answer": answer,
            "time": inference_time
        })
    return results
