# ============================================================
# rag_variants.py
# Contains all RAG variants used for evaluation
# With: Retry, Backoff, Model Fallback for Gemini
# ============================================================

import os
import time
import random

from google import genai
from google.genai.errors import ServerError, ClientError

# Initialize Gemini client
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ============================================================
# GEMINI SAFE CALL WRAPPER (Retry + Backoff + Fallback)
# ============================================================

PRIMARY_MODEL = "models/gemini-2.5-flash-preview-05-20"
FALLBACK_MODEL = "models/gemini-2.0-flash"  # more stable


def safe_gemini_call(prompt, max_retries=5):
    """
    Makes a Gemini API call with:
    - automatic retry on 503 errors
    - exponential backoff with jitter
    - fallback to a stable model
    """
    for attempt in range(1, max_retries + 1):
        try:
            return gemini_client.models.generate_content(
                model=PRIMARY_MODEL,
                contents=prompt
            )

        except (ServerError, ClientError) as e:
            print(f"⚠️ Gemini error on attempt {attempt}: {e}")

            # Backoff time: exponential + random jitter
            delay = min(2 ** attempt, 10) + random.uniform(0, 1)
            print(f"⏳ Retrying in {delay:.2f} seconds...")
            time.sleep(delay)

    # If all retries fail → fallback
    print(f"⚠️ Switching to fallback model: {FALLBACK_MODEL}")
    return gemini_client.models.generate_content(
        model=FALLBACK_MODEL,
        contents=prompt
    )


# ============================================================
# 1. SIMPLE RAG
# ============================================================

def rag_simple(query, retriever, top_k=3):
    """Basic RAG: retrieve top-k context, answer using Gemini."""
    results = retriever.retrieve(query, top_k=top_k)
    context = "\n\n".join([doc["content"] for doc in results]) if results else ""

    if not context:
        return "No relevant context found."

    prompt = f"""
Use ONLY the following context to answer concisely.

Context:
{context}

Question: {query}

Answer:
"""

    response = safe_gemini_call(prompt)
    return response.text.strip()


# ============================================================
# 2. RAG WITH RL OPTIMIZER
# ============================================================

def rag_with_optimizer(query, retriever, optimizer):
    """RAG using dynamic k from RL policy."""
    k = optimizer.get_optimal_k(query)
    results = retriever.retrieve(query, top_k=k)
    context = "\n\n".join([doc["content"] for doc in results]) if results else ""

    if not context:
        return "No relevant context found."

    prompt = f"""
Use ONLY the following context to answer concisely.

Context:
{context}

Question: {query}

Answer:
"""

    response = safe_gemini_call(prompt)
    return response.text.strip()


# ============================================================
# 3. ADVANCED RAG (with metadata)
# ============================================================

def rag_advanced(query, retriever, top_k=5, min_score=0.2, return_context=False):
    """RAG with metadata, sources, and confidence."""
    results = retriever.retrieve(query, top_k=top_k, score_threshold=min_score)

    if not results:
        return {
            "answer": "No relevant context found.",
            "sources": [],
            "confidence": 0.0,
            "context": "" if return_context else None
        }

    context = "\n\n".join([doc["content"] for doc in results])
    sources = [{
        "source": doc["metadata"].get("source_file", "unknown"),
        "page": doc["metadata"].get("page", "unknown"),
        "score": doc["similarity_score"],
        "preview": doc["content"][:300] + "..."
    } for doc in results]

    confidence = max(doc["similarity_score"] for doc in results)

    prompt = f"""
Use ONLY the following context to answer concisely.

Context:
{context}

Question: {query}

Answer:
"""

    response = safe_gemini_call(prompt)
    answer = response.text.strip()

    output = {
        "answer": answer,
        "sources": sources,
        "confidence": confidence
    }

    if return_context:
        output["context"] = context

    return output


# ============================================================
# 4. RAG WITH HALLUCINATION CONTROL
# ============================================================

def rag_with_hallucination_control(query, retriever, hallucination_detector, top_k=5):
    """RAG with hallucination detection + regeneration."""
    results = retriever.retrieve(query, top_k=top_k)

    if not results:
        return {
            "final_answer": "No relevant context found.",
            "hallucination_result": {"is_grounded": False}
        }

    context = "\n\n".join([doc["content"] for doc in results])
    chunks = [doc["content"] for doc in results]

    # Initial Answer
    prompt = f"""
Use ONLY the following context to answer concisely.

Context:
{context}

Question: {query}

Answer:
"""
    response = safe_gemini_call(prompt)
    initial_answer = response.text.strip()

    hall = hallucination_detector.detect(initial_answer, chunks)

    # If hallucinated → regenerate grounded answer
    if not hall["is_grounded"]:
        regen_prompt = f"""
Answer ONLY using the context below.
If answer is not present, reply exactly:
'Information not found in the context.'

Context:
{context}

Question: {query}

Answer:
"""
        regen_response = safe_gemini_call(regen_prompt)
        final_answer = regen_response.text.strip()
    else:
        final_answer = initial_answer

    return {
        "initial_answer": initial_answer,
        "final_answer": final_answer,
        "hallucination_result": hall
    }


# ============================================================
# 5. RAG WITH RL + HALLUCINATION CONTROL
# ============================================================

def rag_rl_with_hc(query, retriever, optimizer, hallucination_detector):
    """RL-selected k + hallucination control."""
    k = optimizer.get_optimal_k(query)
    results = retriever.retrieve(query, top_k=k)

    if not results:
        return {
            "final_answer": "No relevant context found.",
            "hallucination_result": {"is_grounded": False}
        }

    context = "\n\n".join([doc["content"] for doc in results])
    chunks = [doc["content"] for doc in results]

    # Initial Answer
    prompt = f"""
Use ONLY the following context to answer concisely.

Context:
{context}

Question: {query}

Answer:
"""
    response = safe_gemini_call(prompt)
    initial_answer = response.text.strip()

    hall = hallucination_detector.detect(initial_answer, chunks)

    # Regenerate if hallucinated
    if not hall["is_grounded"]:
        regen_prompt = f"""
Answer ONLY using the context below.
If answer is not present, reply exactly:
'Information not found in the context.'

Context:
{context}

Question: {query}

Answer:
"""
        regen_response = safe_gemini_call(regen_prompt)
        final_answer = regen_response.text.strip()
    else:
        final_answer = initial_answer

    return {
        "initial_answer": initial_answer,
        "final_answer": final_answer,
        "hallucination_result": hall
    }
