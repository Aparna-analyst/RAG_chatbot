
# Enterprise RAG Chatbot with Hallucination Control + RL Optimization  
*A Retrieval-Augmented Generation (RAG) system built for Digital University Kerala documents*

---

##  Overview

This project implements a complete end-to-end **enterprise-grade RAG system** powered by:

- PDF ingestion + chunking  
- ChromaDB vector store  
- Sentence-BERT embeddings  
- Gemini LLM  
- Reinforcement Learning–based retrieval optimization  
- Hallucination detection + grounded regeneration  
- A Streamlit chatbot interface  
- A complete evaluation framework comparing 5 RAG variants

The system answers queries using DUK’s official documents such as the **Prospectus 2025** and **PG Regulations 2023**.

---

##  Key Features

###  RAG with PDF Knowledge Base
- Extracts text using PyPDFLoader  
- Splits content into overlapping semantic chunks  
- Embeds using `all-MiniLM-L6-v2`  
- Stores vectors in ChromaDB  
- Retrieves the most relevant chunks for each query  

### Hallucination Detection + Control
- Cosine similarity check between answer and retrieved context  
- Optional NLI (entailment/contradiction) check  
- Auto-regeneration of answers using “Only answer from context” rule  
- Prevents fabricated or unsafe responses  

###  RL-Based Retrieval Optimizer
- Learns optimal `k` (top-k chunks) per query  
- Reward = semantic quality − token cost  
- Supports incremental batch training  
- Saves trained policy to disk  

### Evaluation Framework
Compares **five** RAG models:

1. Simple RAG  
2. RL-Optimized RAG  
3. Advanced RAG (citations, summaries)  
4. Hallucination-Controlled RAG  
5. RL + Hallucination Control  

Metrics used:
- Semantic accuracy (cosine similarity)
- Hallucination rate
- Response time
- Qualitative answer relevance

Outputs: CSV + graphs (accuracy bar chart, hallucination chart)

---

## Architecture

```

PDFs → Chunking → Embeddings → ChromaDB → Retriever
↓
RAG Engine
┌────────────────────────────────────────────────────────────────┐
│ Simple RAG | RL RAG | Advanced RAG | HC RAG | RL + HC RAG     │
└────────────────────────────────────────────────────────────────┘
↓
Hallucination Detection (optional)
↓
Streamlit Chatbot UI

````

---

##  Evaluation Summary

###  Accuracy
- **Advanced RAG** had the highest raw semantic similarity  
  (because it produces longer answers similar to the long gold reference answers).

- **Hallucination-Controlled RAG** had **balanced accuracy + high safety**  
  (shorter, grounded answers → slightly lower similarity score).

### Hallucination Rate
- **HC models drastically reduced hallucination**, especially on missing-context queries.
- RL + HC produced fewer hallucinations but lower accuracy (due to low k values).

###  Deployment Choice
**RAG + Hallucination Control** was selected for deployment because it gives:

- Reliable answers  
- Zero hallucinations in missing-context cases  
- Good accuracy  
- Safe enterprise behavior  

---

##  Visualizations

The evaluation includes two key charts that summarize model performance across all RAG variants.

### **1.Accuracy Comparison of RAG Variants**

This bar chart shows the **average semantic accuracy** (cosine similarity) for each method:

- Simple RAG  
- RL Optimizer RAG  
- Advanced RAG  
- Hallucination-Controlled RAG  
- RL + Hallucination Control  

#### Chart:
<img width="989" height="490" alt="image" src="https://github.com/user-attachments/assets/4fc8238d-5276-425c-9b8c-64412105024b" />


> This chart highlights that **Advanced RAG** produces the highest raw similarity due to longer answers, while **Hallucination-Controlled RAG** provides balanced accuracy with grounded, safe responses.

---

### **2.Hallucination Rate Comparison**

This chart shows how often each method produced a hallucinated answer  
(according to the groundedness detection system).

####  Chart:
<img width="789" height="490" alt="image" src="https://github.com/user-attachments/assets/f503fab5-f55f-46d4-abd8-40151e62d34d" />


> Hallucination-Controlled models perform significantly better in preventing hallucinated content, especially in missing-context scenarios.

---

## RAG Variants Implemented

### 1.Simple RAG
Top-k retrieval → LLM answer generation.

### 2. RL Optimized RAG
Learns best value of `k` for each query.

### 3. Advanced RAG
Includes:
- Structured citations  
- Context preview  
- Confidence score  
- Optional summaries  

### 4. Hallucination-Controlled RAG
- Regenerates answer if similarity < threshold  
- Enforces context-grounded output  

### 5. RL + HC RAG
Combination of optimal retrieval + hallucination-safe answering.

---

## Future Scope

### **A. Debug Retriever (Missing Context Fix)**
- Improve chunking strategy  
- Validate extracted pages  
- Fix inconsistent PDF parsing  

### **B. Improve RL Optimizer**
- Increase training queries  
- Increase `max_k`  
- Use gold-answer-based reward shaping  

### **C. More Visual Graphs**
- Spider/radar charts  
- Heatmaps  
- Error distribution charts  

### **D. Project Report Evaluation Section**
- Automated statistical summary  
- Paper-style tables  
- Comparative analysis write-up  

### **E. Summarize Gold Answers**
- Normalize reference answers for fairer semantic scoring   

---

##  Installation

### 1. Create virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # macOS/Linux
````

### 2. Install packages

```bash
pip install -r requirements.txt
```

### 3. Environment variables

Create `.env`:

```
GEMINI_API_KEY=your_api_key_here
```

---

##  Running the App

### Streamlit App

```bash
streamlit run app.py
```

### RL Training

```bash
python rl_training.py
```

### Evaluation

```bash
python evaluation.py
```

---

##  Acknowledgements

This project uses:

* Google Gemini API
* LangChain
* Sentence Transformers
* ChromaDB
* Streamlit
* NumPy, Pandas, Matplotlib



