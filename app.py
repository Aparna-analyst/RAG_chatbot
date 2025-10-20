import streamlit as st
from utils import rag_retriever, llm, rag_with_hallucination_control, hallucination_detector
import pandas as pd

# ------------------- Page Config ------------------- #
st.set_page_config(page_title="Enterprise RAG Chatbot", layout="wide")
st.title("📚 Enterprise RAG Chatbot")

# ------------------- Initialize session state ------------------- #
if 'history' not in st.session_state:
    st.session_state.history = []
if 'selected_query' not in st.session_state:
    st.session_state.selected_query = ""
if 'selected_result' not in st.session_state:
    st.session_state.selected_result = None

# ------------------- Sidebar: Query History ------------------- #
st.sidebar.header("Query History")

if st.sidebar.button("🗑️ Clear History"):
    st.session_state.history.clear()
    st.session_state.selected_query = ""
    st.session_state.selected_result = None
    st.rerun()

if st.session_state.history:
    for i, item in enumerate(reversed(st.session_state.history)):
        query = item["query"]
        status = item["hallucination_result"]["status"]
        if st.sidebar.button(f"{query[:50]}... | {status}", key=f"q_{i}"):
            st.session_state.selected_query = query
            st.session_state.selected_result = item

# ------------------- User Input ------------------- #
st.subheader("Ask a Question")

query = st.text_area("Enter your question:", value=st.session_state.selected_query)

if st.button("Submit") and query.strip():
    with st.spinner("Retrieving and generating answer..."):
        result = rag_with_hallucination_control(query, rag_retriever, llm, hallucination_detector)

    st.session_state.history.append(result)
    st.session_state.selected_query = query
    st.session_state.selected_result = result

# ------------------- Display Current Answer ------------------- #
if st.session_state.selected_result:
    result = st.session_state.selected_result

    st.markdown("### 💬 Answer")
    st.write(result["final_answer"])

    st.markdown("### 🔍 Hallucination Detection Results")
    hall = result["hallucination_result"]
    nli_label = hall["nli_result"]["label"] if hall["nli_result"] else "N/A"
    nli_score = hall["nli_result"]["score"] if hall["nli_result"] else "N/A"

    hall_df = pd.DataFrame({
        "Field": ["Similarity", "Grounded", "Needs Regeneration", "Status", "NLI Label", "NLI Score"],
        "Value": [
            hall.get("similarity", None),
            hall.get("is_grounded", None),
            hall.get("needs_regeneration", None),
            hall.get("status", None),
            nli_label,
            nli_score
        ]
    })
    st.table(hall_df)
