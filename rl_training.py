import os
import json
import pickle
import time
import numpy as np
from google import genai
from google.genai import types
from google.genai import errors

# ------------------- Load environment ------------------- #
from dotenv import load_dotenv
load_dotenv()

gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# =========================================================
# SAFE GENERATION WITH RETRY + HEARTBEAT
# =========================================================
def safe_generate(model_name, prompt, max_retries=5):
    wait_time = 5  # starting backoff
    
    for attempt in range(max_retries):
        try:
            return gemini_client.models.generate_content(
                model=model_name,
                contents=prompt
            )
        except Exception as e:
            error_text = str(e)

            if "RESOURCE_EXHAUSTED" in error_text or "429" in error_text:
                print(f"\n⚠️ 429 Rate limit hit (attempt {attempt+1}/{max_retries})")

                # Heartbeat during wait
                for i in range(wait_time):
                    print(f"⏳ Cooling down... {wait_time - i}s remaining (heartbeat)", flush=True)
                    time.sleep(1)

                wait_time = min(wait_time * 2, 30)
            else:
                raise e

    print("❌ Max retries exceeded. Skipping this call.")
    return None


# =========================================================
# RETRIEVAL OPTIMIZER (RL)
# =========================================================
class RetrievalOptimizer:
    def __init__(self, retriever, max_k=3):
        self.retriever = retriever
        self.max_k = max_k
        self.policy = {}

    def reward_function(self, quality, tokens):
        return quality - 0.001 * tokens

    def evaluate_response(self, answer, question):
        emb_q = self.retriever.embedding_manager.generate_embeddings([question])[0]
        emb_a = self.retriever.embedding_manager.generate_embeddings([answer])[0]
        return float(np.dot(emb_q, emb_a) /
                     (np.linalg.norm(emb_q) * np.linalg.norm(emb_a)))

    def train_batch(self, batch_queries, batch_number):
        print(f"\n\n====================")
        print(f"🚀 STARTING BATCH {batch_number}")
        print(f"====================\n")

        for query in batch_queries:
            print(f"\n🔎 Query: {query}")

            best_k, best_reward = 3, -np.inf

            for k in range(1, self.max_k + 1):
                print(f"➡️ Trying k={k}")

                results = self.retriever.retrieve(query, top_k=k)
                if not results:
                    print(f"⚠️ No documents retrieved for k={k}")
                    continue

                context = "\n\n".join([r['content'] for r in results])

                prompt = (
                    f"Use the context below to answer:\n{context}\n\n"
                    f"Question: {query}\nAnswer:"
                )

                response = safe_generate("models/gemini-2.5-flash-lite", prompt)

                if response is None:
                    print(f"⚠️ Skipping k={k} after retries")
                    continue

                answer_text = response.text
                quality = self.evaluate_response(answer_text, query)
                reward = self.reward_function(quality, len(prompt.split()))

                print(f"   ✔ Reward = {reward:.4f}")

                if reward > best_reward:
                    best_reward = reward
                    best_k = k

                time.sleep(0.5)

            self.policy[query] = best_k
            print(f"🎉 Selected k={best_k} for this query")

        print(f"\n🏁 Completed Batch {batch_number}!")


# =========================================================
# CHUNKING
# =========================================================
def chunk_queries(queries, size=10):
    for i in range(0, len(queries), size):
        yield queries[i:i + size]


# =========================================================
# MAIN TRAINING ENTRY POINT
# =========================================================
def run_batches():
    # Import retriever from your project
    from utils import rag_retriever

    # Load training data
    with open("data/duk_training_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    training_queries = [q["query"] for q in data]

    # Prepare optimizer
    optimizer_path = "data/reward_policy_gemini.pkl"
    optimizer = RetrievalOptimizer(rag_retriever, max_k=3)

    if os.path.exists(optimizer_path):
        print("📄 Loading existing policy...")
        optimizer.policy = pickle.load(open(optimizer_path, "rb"))
    else:
        print("🆕 No existing policy found. Starting fresh...")

    # Create batches
    batches = list(chunk_queries(training_queries, size=10))

    # Run each batch
    for i, batch in enumerate(batches, start=1):
        optimizer.train_batch(batch, i)

        # Save after each batch
        with open(optimizer_path, "wb") as f:
            pickle.dump(optimizer.policy, f)

        print(f"💾 Saved policy after Batch {i}\n")
        print("⏸ You may stop now and run again later.\n")

        # Optional rest
        time.sleep(5)


if __name__ == "__main__":
    run_batches()
