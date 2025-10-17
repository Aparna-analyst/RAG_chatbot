import streamlit as st
from utils import rag_retriever, llm, rag_with_optimizer, retrieval_optimizer, hallucination_detector
import pandas as pd

# ------------------- Page Config ------------------- #
st.set_page_config(page_title="Enterprise RAG Chatbot", layout="wide")
st.title("📚 Enterprise RAG Chatbot")

# ------------------- Initialize session state ------------------- #
if 'history' not in st.session_state:
    st.session_state.history = []
if 'selected_query' not in st.session_state:
    st.session_state.selected_query = ""
if 'selected_answer' not in st.session_state:
    st.session_state.selected_answer = ""
if 'selected_hall' not in st.session_state:
    st.session_state.selected_hall = None

# ------------------- Sidebar: Interactive Query History ------------------- #
st.sidebar.header("Query History")
if st.session_state.history:
    for i, (q, a, hall) in enumerate(reversed(st.session_state.history)):
        # Use a button for each query
        if st.sidebar.button(f"{q[:50]}... | {hall['status']}", key=i):
            st.session_state.selected_query = q
            st.session_state.selected_answer = a
            st.session_state.selected_hall = hall

# ------------------- User Input ------------------- #
st.subheader("Ask a Question")

# If a past query is selected, pre-fill the text area
query = st.text_area("Enter your question:", value=st.session_state.selected_query)

if st.button("Submit") and query.strip():
    # RL-optimized RAG retrieval
    answer = rag_with_optimizer(query, rag_retriever, llm, optimizer=retrieval_optimizer)
    
    # Hallucination detection
    hall_result = hallucination_detector.detect(answer, [])

    # Store in history
    st.session_state.history.append((query, answer, hall_result))

    # Update selected query to current
    st.session_state.selected_query = query
    st.session_state.selected_answer = answer
    st.session_state.selected_hall = hall_result

# ------------------- Display Current Answer ------------------- #
if st.session_state.selected_answer:
    st.markdown("**Answer:**")
    st.write(st.session_state.selected_answer)

    st.markdown("**Hallucination Detection:**")
    hall = st.session_state.selected_hall
    hall_df = pd.DataFrame({
        "Field": ["Similarity", "Grounded", "Needs Regeneration", "Status", "NLI Label", "NLI Score"],
        "Value": [
            hall["similarity"],
            hall["is_grounded"],
            hall["needs_regeneration"],
            hall["status"],
            hall["nli_result"]["label"] if hall["nli_result"] else None,
            hall["nli_result"]["score"] if hall["nli_result"] else None
        ]
    })
    st.table(hall_df)
