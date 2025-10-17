import streamlit as st
from utils import rag_with_hallucination_control, rag_retriever, llm, hallucination_detector
from pathlib import Path

# ------------------- Page Config ------------------- #
st.set_page_config(page_title="Enterprise RAG Chatbot", layout="wide")
st.title("📚 Enterprise RAG Chatbot with Hallucination Control & RL Retrieval")

# ------------------- Sidebar ------------------- #
st.sidebar.header("Query History")
if "history" not in st.session_state:
    st.session_state.history = []

history_expander = st.sidebar.expander("Previous Queries", expanded=True)
for idx, item in enumerate(reversed(st.session_state.history)):
    st.markdown(f"**Q:** {item['query']}")
    st.markdown(f"**A:** {item['answer'][:300]}...")  # preview only
    st.markdown("---")

# ------------------- User Input ------------------- #
user_query = st.text_area("Enter your question here:")

top_k = st.slider("Select top-k retrieval documents", min_value=1, max_value=10, value=5)

if st.button("Get Answer"):
    if user_query.strip() == "":
        st.warning("Please enter a query!")
    else:
        with st.spinner("Generating response..."):
            result = rag_with_hallucination_control(
                query=user_query,
                retriever=rag_retriever,
                llm=llm,
                hallucination_detector=hallucination_detector,
                top_k=top_k
            )

        # ------------------- Display Output ------------------- #
        st.markdown("### 🧾 Final Answer")
        st.write(result["final_answer"])

        st.markdown("### 🔍 Hallucination Check")
        st.json(result["hallucination_result"])

        # Expanders for details
        with st.expander("Show Initial Answer"):
            st.write(result["initial_answer"])

        with st.expander("Show Retrieved Sources"):
            retrieved_chunks = result.get("hallucination_result", {}).get("retrieved_chunks", [])
            if retrieved_chunks:
                for idx, chunk in enumerate(retrieved_chunks):
                    st.markdown(f"**Source {idx+1}:** {chunk[:300]}...")
            else:
                st.write("No sources retrieved.")

        # Save to session history
        st.session_state.history.append({
            "query": user_query,
            "answer": result["final_answer"]
        })
