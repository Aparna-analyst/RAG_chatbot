# rl_optimizer.py
# Contains only the RL RetrievalOptimizer class
# Clean version for evaluation + training reuse.

import numpy as np

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
        return float(
            np.dot(emb_q, emb_a) /
            (np.linalg.norm(emb_q) * np.linalg.norm(emb_a))
        )

    def get_optimal_k(self, query):
        """Return optimal k, fallback = mean policy."""
        if query in self.policy:
            return self.policy[query]

        if len(self.policy) == 0:
            return 3  # default if no training

        return int(np.mean(list(self.policy.values())))
