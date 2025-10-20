import streamlit as st
from utils import rag_retriever, llm, rag_with_hallucination_control, hallucination_detector
import pandas as pd

# ------------------- Page Config ------------------- #
st.set_page_config(page_title="Enterprise RAG Chatbot", layout="wide")
st.title("🤖 Enterprise RAG Chatbot")

# ------------------- Initialize session state ------------------- #
if "history" not in st.session_state:
    st.session_state.history = []

# ------------------- Sidebar: Query History ------------------- #
st.sidebar.header("🕒 Query History")

if st.sidebar.button("🗑️ Clear All History"):
    st.session_state.history.clear()
    st.rerun()

# Display history in sidebar (most recent first)
if st.session_state.history:
    for i, item in enumerate(reversed(st.session_state.history)):
        query = item["query"]
        status = item["hallucination_result"]["status"]
        if st.sidebar.button(f"{query[:40]}... | {status}", key=f"hist_{i}"):
            st.session_state.selected_index = len(st.session_state.history) - 1 - i
            st.rerun()

# ------------------- Display Chat Conversation ------------------- #
for idx, chat in enumerate(st.session_state.history):
    with st.chat_message("user"):
        st.markdown(f"**{chat['query']}**")

    with st.chat_message("assistant"):
        st.write(chat["final_answer"])

        hall = chat["hallucination_result"]
        nli_label = hall["nli_result"]["label"] if hall["nli_result"] else "N/A"
        nli_score = hall["nli_result"]["score"] if hall["nli_result"] else "N/A"

        with st.expander("🔍 Hallucination Detection Details"):
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

# ------------------- User Input ------------------- #
query = st.chat_input("Type your question here...")

if query:
    with st.chat_message("user"):
        st.markdown(f"**{query}**")

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating answer..."):
            result = rag_with_hallucination_control(query, rag_retriever, llm, hallucination_detector)

        st.write(result["final_answer"])

        hall = result["hallucination_result"]
        nli_label = hall["nli_result"]["label"] if hall["nli_result"] else "N/A"
        nli_score = hall["nli_result"]["score"] if hall["nli_result"] else "N/A"

        with st.expander("🔍 Hallucination Detection Details"):
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

    # Save query and result to history
    st.session_state.history.append({
        "query": query,
        **result
    })
