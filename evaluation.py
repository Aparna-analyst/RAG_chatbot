# ============================================================
# evaluation.py
# Compare 5 RAG Variants on:
#   - Accuracy (semantic similarity)
#   - Hallucination rate
# ============================================================

import os
import json
import time
import csv
from pathlib import Path

from sentence_transformers import SentenceTransformer, util

# --- Import RAG variants ---
from rag_variants import (
    rag_simple,
    rag_with_optimizer,
    rag_advanced,
    rag_with_hallucination_control,
    rag_rl_with_hc
)

# --- Import retriever and hallucination detector ---
from utils import rag_retriever, hallucination_detector

# ============================================================
# FIXED: Import RL Optimizer class
# ============================================================

from rl_optimizer import RetrievalOptimizer  # <-- NEW & IMPORTANT

# --- Load RL Optimizer policy dict ---
import pickle
optimizer_path = "data/reward_policy_gemini.pkl"

optimizer = None
if os.path.exists(optimizer_path):
    print("📄 Loading trained RL policy...")
    with open(optimizer_path, "rb") as f:
        saved_policy = pickle.load(f)

    # Rebuild proper RL object
    optimizer = RetrievalOptimizer(retriever=rag_retriever, max_k=3)
    optimizer.policy = saved_policy  # attach the learned policy dict
    print("✅ RL Optimizer loaded successfully.")

else:
    print("⚠️ RL optimizer policy not found! RL-based variants will be skipped.")

# ============================================================
# CONFIG PATHS
# ============================================================

TEST_QUERIES_PATH = r"C:\Users\aparn\Downloads\Gen AI project\data\duk_test_data.json"
GOLD_ANSWERS_PATH = r"C:\Users\aparn\Downloads\Gen AI project\data\duk_gold_answers.json"
CSV_OUTPUT_PATH = "evaluation_results.csv"

# ============================================================
# ACCURACY MODEL
# ============================================================

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def compute_similarity(a, b):
    """Return cosine similarity between answer and gold answer."""
    emb_a = embedding_model.encode(a, convert_to_tensor=True)
    emb_b = embedding_model.encode(b, convert_to_tensor=True)
    return round(util.cos_sim(emb_a, emb_b).item(), 4)

# ============================================================
# UTILS
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ============================================================
# MAIN EVALUATION
# ============================================================

def evaluate():
    print("\n🔍 Loading test queries...")
    test_queries = load_json(TEST_QUERIES_PATH)

    print("🔍 Loading gold answers...")
    gold_data = load_json(GOLD_ANSWERS_PATH)

    # Convert list -> dict
    gold_dict = {item["query"]: item["gold_answer"] for item in gold_data}

    results = []

    print("\n🚀 Starting comparison of all RAG variants...\n")

    for item in test_queries:
        query = item["query"]
        gold_answer = gold_dict.get(query, "")

        if not gold_answer:
            print(f"⚠️ Missing gold answer for query: {query}")
            continue

        print(f"\n======================================================")
        print(f"🔎 QUERY: {query}")
        print(f"======================================================")

        row = {"query": query, "gold_answer": gold_answer}

        # ---------------- SIMPLE RAG ----------------
        print("\n➡️ Simple RAG")
        start = time.time()
        simple_ans = rag_simple(query, rag_retriever)
        row["simple_answer"] = simple_ans
        row["simple_accuracy"] = compute_similarity(simple_ans, gold_answer)
        row["simple_hallucination"] = None
        row["simple_time"] = round(time.time() - start, 3)

        # ---------------- RL OPTIMIZER RAG ----------------
        print("\n➡️ RAG + RL Optimizer")
        if optimizer:
            start = time.time()
            rl_ans = rag_with_optimizer(query, rag_retriever, optimizer)
            row["rl_answer"] = rl_ans
            row["rl_accuracy"] = compute_similarity(rl_ans, gold_answer)
            row["rl_hallucination"] = None
            row["rl_time"] = round(time.time() - start, 3)
        else:
            row["rl_answer"] = "Optimizer not loaded"
            row["rl_accuracy"] = None
            row["rl_hallucination"] = None

        # ---------------- ADVANCED RAG ----------------
        print("\n➡️ Advanced RAG")
        start = time.time()
        adv_out = rag_advanced(query, rag_retriever)
        adv_ans = adv_out["answer"]
        row["advanced_answer"] = adv_ans
        row["advanced_accuracy"] = compute_similarity(adv_ans, gold_answer)
        row["advanced_hallucination"] = None
        row["advanced_time"] = round(time.time() - start, 3)

        # ---------------- HALLUCINATION CONTROL ----------------
        print("\n➡️ RAG + Hallucination Control")
        start = time.time()
        hc_out = rag_with_hallucination_control(query, rag_retriever, hallucination_detector)
        hc_ans = hc_out["final_answer"]
        row["hc_answer"] = hc_ans
        row["hc_accuracy"] = compute_similarity(hc_ans, gold_answer)
        row["hc_hallucination"] = not hc_out["hallucination_result"]["is_grounded"]
        row["hc_time"] = round(time.time() - start, 3)

        # ---------------- RL + HALLUCINATION CONTROL ----------------
        print("\n➡️ RAG + RL + Hallucination Control")
        if optimizer:
            start = time.time()
            rl_hc = rag_rl_with_hc(query, rag_retriever, optimizer, hallucination_detector)
            rl_hc_ans = rl_hc["final_answer"]
            row["rl_hc_answer"] = rl_hc_ans
            row["rl_hc_accuracy"] = compute_similarity(rl_hc_ans, gold_answer)
            row["rl_hc_hallucination"] = not rl_hc["hallucination_result"]["is_grounded"]
            row["rl_hc_time"] = round(time.time() - start, 3)
        else:
            row["rl_hc_answer"] = "Optimizer not loaded"
            row["rl_hc_accuracy"] = None
            row["rl_hc_hallucination"] = None

        results.append(row)

    # ---------------- SAVE CSV ----------------
    print("\n💾 Saving CSV:", CSV_OUTPUT_PATH)

    with open(CSV_OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(results[0].keys())
        for r in results:
            writer.writerow(r.values())

    print("\n🎉 Evaluation Complete!")
    print(f"Results saved at: {CSV_OUTPUT_PATH}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    evaluate()
