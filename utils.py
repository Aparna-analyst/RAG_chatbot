import os
from pathlib import Path
import uuid
import shutil
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import pickle

# -------------------- Load API Key -------------------- #
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# -------------------- Data Ingestion -------------------- #
def process_all_pdfs(pdf_directory: Path):
    all_documents = []
    pdf_files = list(pdf_directory.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDFs in {pdf_directory}")

    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_file))
            documents = loader.load()
            for doc in documents:
                doc.metadata['source_file'] = pdf_file.name
                doc.metadata['file_type'] = 'pdf'
            all_documents.extend(documents)
        except Exception as e:
            print(f"⚠️ Failed to process {pdf_file.name}: {e}")
    return all_documents

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(documents)

# -------------------- Embedding & Vector Store -------------------- #
class EmbeddingManager:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, texts):
        print(f"Generating embeddings for {len(texts)} chunks...")
        return self.model.encode(texts, show_progress_bar=True)

class VectorStore:
    def __init__(self, collection_name="pdf_docs", persist_directory=None):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        os.makedirs(self.persist_directory, exist_ok=True)

        try:
            self.client = chromadb.PersistentClient(path=str(self.persist_directory))
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "PDF embeddings for RAG"}
            )
        except Exception as e:
            print(f"⚠️ ChromaDB error: {e}\nRebuilding vector store...")
            shutil.rmtree(self.persist_directory, ignore_errors=True)
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(self.persist_directory))
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Rebuilt PDF embeddings for RAG"}
            )

    def add_documents(self, documents, embeddings):
        ids, metadatas, docs_text, embeddings_list = [], [], [], []
        for i, (doc, emb) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)
            metadatas.append({**doc.metadata, 'doc_index': i, 'content_length': len(doc.page_content)})
            docs_text.append(doc.page_content)
            embeddings_list.append(emb.tolist())

        self.collection.add(ids=ids, embeddings=embeddings_list, metadatas=metadatas, documents=docs_text)
        print(f"✅ Added {len(documents)} documents to ChromaDB.")

# -------------------- RAG Retriever -------------------- #
class RAGRetriever:
    def __init__(self, vector_store, embedding_manager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query, top_k=5, score_threshold=0.0):
        query_emb = self.embedding_manager.generate_embeddings([query])[0]
        results = self.vector_store.collection.query(query_embeddings=[query_emb.tolist()], n_results=top_k)
        retrieved_docs = []
        if results['documents'] and results['documents'][0]:
            for i, (doc_id, doc, meta, dist) in enumerate(zip(
                results['ids'][0],
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                similarity_score = 1 - dist
                if similarity_score >= score_threshold:
                    retrieved_docs.append({
                        'id': doc_id,
                        'content': doc,
                        'metadata': meta,
                        'similarity_score': similarity_score,
                        'distance': dist,
                        'rank': i + 1
                    })
        return retrieved_docs

# -------------------- LLM -------------------- #
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",
    temperature=0.1,
    max_tokens=1024
)

# -------------------- Hallucination Detection -------------------- #
class HallucinationDetector:
    def __init__(self, model_name="all-MiniLM-L6-v2", nli_model="facebook/bart-large-mnli",
                 threshold=0.6, nli_trigger=0.7):
        self.model = SentenceTransformer(model_name)
        self.nli = pipeline("text-classification", model=nli_model)
        self.threshold = threshold
        self.nli_trigger = nli_trigger

    def compute_similarity(self, response, retrieved_chunks):
        context_text = " ".join(retrieved_chunks)
        resp_emb = self.model.encode(response, convert_to_tensor=True)
        ctx_emb = self.model.encode(context_text, convert_to_tensor=True)
        return util.cos_sim(resp_emb, ctx_emb).item()

    def nli_check(self, response, retrieved_chunks):
        context_text = " ".join(retrieved_chunks)[:3000]
        combined_text = f"{context_text} </s></s> {response}"
        result = self.nli(combined_text)[0]
        return {"label": result.get("label", "ERROR"), "score": round(result.get("score", 0.0), 3)}

    def detect(self, response, retrieved_chunks):
        similarity = self.compute_similarity(response, retrieved_chunks)
        nli_info = None
        is_grounded = similarity >= self.threshold
        needs_regeneration = False

        if similarity < self.nli_trigger:
            nli_info = self.nli_check(response, retrieved_chunks)
            if nli_info["label"].upper() == "ENTAILMENT" and nli_info["score"] > 0.7:
                is_grounded = True
            else:
                needs_regeneration = True

        status = "Grounded ✅" if is_grounded else "Hallucination ⚠️"
        return {
            "similarity": round(similarity, 3),
            "nli_result": nli_info,
            "is_grounded": is_grounded,
            "needs_regeneration": needs_regeneration,
            "status": status
        }

# -------------------- RL Optimizer Wrapper -------------------- #
class RetrievalOptimizerWrapper:
    def __init__(self, policy_dict):
        self.policy = policy_dict

    def get_optimal_k(self, query):
        # Example: return stored default_k or implement actual policy logic
        return self.policy.get("default_k", 5)

# -------------------- RAG with RL Optimizer -------------------- #
def rag_with_optimizer(query, retriever, llm, optimizer=None, max_k=10):
    if optimizer:
        k = optimizer.get_optimal_k(query)
        k = min(k, max_k)
        print(f"🎯 RL-selected top_k = {k}")
    else:
        k = 5
        print(f"🎯 Using default top_k = {k}")

    retrieved_docs = retriever.retrieve(query, top_k=k)
    context = "\n\n".join([doc['content'] for doc in retrieved_docs]) if retrieved_docs else ""

    if not context:
        return "No relevant context found to answer the question."

    prompt = f"""Use the following context to answer the question concisely.
Context:
{context}

Question: {query}

Answer:"""

    response = llm.invoke([prompt])
    return response.content.strip()

# -------------------- Load RL Optimizer -------------------- #
data_dir = Path(__file__).parent / "data"
rl_policy_path = data_dir / "reward_policy.pkl"

if rl_policy_path.exists():
    with open(rl_policy_path, "rb") as f:
        policy_dict = pickle.load(f)
    retrieval_optimizer = RetrievalOptimizerWrapper(policy_dict)
    print("✅ Loaded RL Retrieval Optimizer")
else:
    retrieval_optimizer = None
    print("⚠️ RL Retrieval Optimizer not found. Using default top_k")

# -------------------- Initialize KB -------------------- #
kb_dir = data_dir / "KB"
vectorstore_dir = data_dir / "vector_store"

all_docs = process_all_pdfs(kb_dir)
chunks = split_documents(all_docs)
embedding_manager = EmbeddingManager()
vectorstore = VectorStore(persist_directory=vectorstore_dir)

if vectorstore.collection.count() == 0:
    print("⚙️ Generating new embeddings...")
    texts = [doc.page_content for doc in chunks]
    embeddings = embedding_manager.generate_embeddings(texts)
    vectorstore.add_documents(chunks, embeddings)
else:
    print(f"✅ Loaded {vectorstore.collection.count()} documents from vector store.")

rag_retriever = RAGRetriever(vectorstore, embedding_manager)
hallucination_detector = HallucinationDetector(threshold=0.6)
