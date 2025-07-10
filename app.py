import streamlit as st
import requests
import faiss
import numpy as np
import pandas as pd
import os
from sentence_transformers import SentenceTransformer

# ✅ Ollama API call
import openai

openai.api_key = st.secrets["OPENAI_API_KEY"]  # Reads API key from Streamlit secrets

def generate_answer(prompt, model_name="gpt-3.5-turbo"):
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful multilingual assistant. Answer in the same language as the question."},
            {"role": "user", "content": prompt}
        ]
    )
    return response["choices"][0]["message"]["content"].strip()


# ✅ FAISS + embedder
@st.cache_resource
def load_embedder_and_index():
    embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    data_dir = "comment_data"
    files = [f"{data_dir}/{f}" for f in os.listdir(data_dir) if f.endswith("_comments.csv")]
    df = pd.concat([pd.read_csv(f) for f in files])
    posts = df["text"].dropna().tolist()

    embeddings = embedder.encode(posts, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    id_to_text = {i: t for i, t in enumerate(posts)}

    return embedder, index, id_to_text

# ✅ RAG-based response with multilingual support
def rag_response(question, embedder, index, id_to_text):
    q_embed = embedder.encode([question], convert_to_numpy=True)
    D, I = index.search(q_embed, 5)
    context = "\n".join([id_to_text[i] for i in I[0]])

    prompt = f"""You are an expert assistant that answers questions based on social media posts in the same language as the question (either English or German).

Context:
{context}

Question: {question}
Answer (in the same language):"""

    answer = generate_answer(prompt)
    return answer, context

# ✅ Streamlit UI
st.set_page_config(page_title="🚗 AMG Chatbot", layout="centered")
st.title("🚗 Mercedes-AMG Feedback Chatbot (Multilingual)")
st.markdown("Ask anything about AMG — in **English** or **German**!")

question = st.text_input("Your question (English or German):")

if st.button("Generate Answer"):
    with st.spinner("Thinking..."):
        embedder, index, id_to_text = load_embedder_and_index()
        answer, context = rag_response(question, embedder, index, id_to_text)

        st.subheader("💬 Answer")
        st.success(answer)

        st.subheader("📄 Retrieved Snippets")
        st.code(context)
