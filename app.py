import streamlit as st
from utils import rag_retriever, llm, rag_with_optimizer, retrieval_optimizer, hallucination_detector

# ------------------- Page Config ------------------- #
st.set_page_config(page_title="Enterprise RAG Chatbot", layout="wide")
st.title("📚 Enterprise RAG Chatbot")

# ------------------- Sidebar ------------------- #
st.sidebar.header("Query History")
if 'history' not in st.session_state:
    st.session_state.history = []

# ------------------- User Input ------------------- #
query = st.text_area("Enter your question:")

if st.button("Submit") and query.strip():
    # Use RL-optimized RAG
    answer = rag_with_optimizer(query, rag_retriever, llm, optimizer=retrieval_optimizer)
    
    # Hallucination detection (optional)
    results = hallucination_detector.detect(answer, [])
    
    st.session_state.history.append((query, answer, results['status']))
    st.success("Answer generated ✅")

# ------------------- Show Query History ------------------- #
if st.session_state.history:
    st.sidebar.subheader("Past Queries")
    for i, (q, a, status) in enumerate(reversed(st.session_state.history)):
        with st.expander(f"Query: {q} | Status: {status}", expanded=False):
            st.write(f"**Answer:** {a}")
