import os
from pathlib import Path
import uuid
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# -------------------- Load API Key -------------------- #
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# -------------------- Data Ingestion -------------------- #
def process_all_pdfs(pdf_directory: Path):
    all_documents = []
    pdf_files = list(pdf_directory.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDFs in {pdf_directory}")

    for pdf_file in pdf_files:
        loader = PyPDFLoader(str(pdf_file))
        documents = loader.load()
        for doc in documents:
            doc.metadata['source_file'] = pdf_file.name
            doc.metadata['file_type'] = 'pdf'
        all_documents.extend(documents)
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
        return self.model.encode(texts, show_progress_bar=True)

class VectorStore:
    def __init__(self, collection_name="pdf_docs", persist_directory=None):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        os.makedirs(self.persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "PDF embeddings for RAG"}
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
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant", temperature=0.1, max_tokens=1024)

# -------------------- Hallucination Detection -------------------- #
class HallucinationDetector:
    def __init__(self, model_name="all-MiniLM-L6-v2", nli_model="facebook/bart-large-mnli",
                 threshold=0.75, nli_trigger=0.6):
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
        return {"similarity": round(similarity,3), "nli_result": nli_info,
                "is_grounded": is_grounded, "needs_regeneration": needs_regeneration,
                "status": status}

# -------------------- Full RAG + Hallucination Pipeline -------------------- #
def rag_with_hallucination_control(query, retriever, llm, hallucination_detector, top_k=5):
    results = retriever.retrieve(query, top_k=top_k)
    if not results:
        return {"query": query, "answer": "No relevant context found."}

    context = "\n\n".join([doc['content'] for doc in results])
    retrieved_chunks = [doc['content'] for doc in results]

    prompt = f"""Use ONLY the context below to answer concisely.
Context:
{context}

Question: {query}

Answer:"""

    response = llm.invoke([prompt])
    answer = response.content.strip()

    hall_result = hallucination_detector.detect(answer, retrieved_chunks)

    # Regenerate if hallucinated
    if not hall_result["is_grounded"]:
        grounded_prompt = f"""Answer ONLY using the context below.
If the answer is not present, reply exactly as: 'Information not found in the context.'

Context:
{context}

Question: {query}

Answer:"""
        grounded_response = llm.invoke([grounded_prompt])
        grounded_answer = grounded_response.content.strip()
    else:
        grounded_answer = answer

    return {"query": query, "initial_answer": answer, "final_answer": grounded_answer, "hallucination_result": hall_result}

# -------------------- Initialize -------------------- #
data_dir = Path(__file__).parent / "data"
kb_dir = data_dir / "KB"
vectorstore_dir = data_dir / "vector_store"

# Process KB PDFs
all_docs = process_all_pdfs(kb_dir)
chunks = split_documents(all_docs)
embedding_manager = EmbeddingManager()

# Initialize vector store
vectorstore = VectorStore(persist_directory=vectorstore_dir)

# Auto-regenerate embeddings if empty
if vectorstore.collection.count() == 0:
    print("Vector store empty. Generating embeddings...")
    texts = [doc.page_content for doc in chunks]
    embeddings = embedding_manager.generate_embeddings(texts)
    vectorstore.add_documents(chunks, embeddings)
else:
    print(f"Vector store already has {vectorstore.collection.count()} documents.")

rag_retriever = RAGRetriever(vectorstore, embedding_manager)
hallucination_detector = HallucinationDetector(threshold=0.75)
