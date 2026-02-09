# Hybrid RAG System

A Retrieval-Augmented Generation system combining dense vector search (ChromaDB), sparse keyword matching (BM25), and cross-encoder re-ranking to answer questions from Wikipedia articles.

**Requirements**: Python 3.10+ • 8GB RAM • 10GB disk space

---

## 1. Hybrid RAG System

### Overview

The system uses a three-stage retrieval pipeline:
1. **Dense retrieval** - Semantic search using embeddings (sentence-transformers/all-mpnet-base-v2)
2. **Sparse retrieval** - Keyword matching using BM25
3. **Re-ranking** - Cross-encoder re-scoring (ms-marco-MiniLM-L-6-v2)
4. **Generation** - Answer synthesis using Qwen 1.5B-Instruct

Data sources: 200 fixed + 300 random Wikipedia articles

### Setup Guide

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running Frontend and Backend

```bash
# Terminal 1: Start backend (FastAPI on port 8000)
python hybrid_rag_app.py

# Terminal 2: Start frontend (Streamlit on port 8501)
streamlit run ui_app.py
```

**First-time setup (via UI)**:
1. Open the Streamlit UI at http://localhost:8501
2. Navigate to **Settings** page (see [frontend/ui_settings_page.py](frontend/ui_settings_page.py))
3. Check "Refresh Fixed" and "Refresh Random" options
4. Click "Start Indexing" (10-20 minutes for ~500 articles)

This triggers data collection from Wikipedia and builds both vector (ChromaDB) and BM25 indices.

### Local Testing Using test_response

```bash
# Test inference without UI
python test_response.py
```

Edit [test_response.py](test_response.py) to modify test queries. Useful for:
- Quick API validation
- Debugging retrieval pipeline
- Performance benchmarking

### Configuration

Edit [src/config/app_config.py](src/config/app_config.py):

```python
# Core models
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# Retrieval parameters
RETRIEVAL_CONFIG = {
    "dense_top_k": 5,        # Dense search results
    "sparse_top_k": 5,       # BM25 results
    "reranker_top_k": 2,     # Final context documents
}

# LLM generation
LLM_CONFIG = {
    "max_new_tokens": 500,
    "temperature": 0.001,
}
```

---

## 2. Evaluation Pipeline

### Overview

Automated evaluation framework for RAG system performance assessment. Uses LLM-generated Q&A pairs with ground truth URLs and measures retrieval accuracy + answer quality.

### QA Generation and Types

The system generates four question types from Wikipedia corpus:

| Type | Description | Example | Distribution |
|------|-------------|---------|--------------|
| **Factual** | Direct fact queries | "What is the capital of France?" | 30% |
| **Comparative** | Comparison questions | "How does X differ from Y?" | 25% |
| **Inferential** | Reasoning/analysis | "Why did X lead to Y?" | 25% |
| **Multi-hop** | Multi-document reasoning | "What's the connection between X and Y?" | 20% |

```bash
# Generate 100 Q&A pairs with default distribution
python run_evaluation.py generate --total-questions 100

# Equal distribution across types
python run_evaluation.py generate --total-questions 100 --equal-distribution
```

### Pipeline Trigger Options

**Option 1: Full pipeline (generate + evaluate)**
```bash
python run_evaluation.py full --total-questions 100
```

**Option 2: Generate only**
```bash
python run_evaluation.py generate --total-questions 100
```

**Option 3: Evaluate with existing dataset**
```bash
python run_evaluation.py evaluate --dataset-file qa_dataset.json
```

**Option 4: Evaluate with limits (faster testing)**
```bash
python run_evaluation.py evaluate \
    --dataset-file qa_dataset.json \
    --max-answer-evaluations 10 \
    --max-retrieval-evaluations 20
```

**Option 5: Quick test (limited generation + evaluation)**
```bash
python run_evaluation.py full \
    --total-questions 20 \
    --max-eval-questions 10
```

### Metrics

#### Retrieval Performance

| Metric | Description | Good Score |
|--------|-------------|------------|
| **MRR** (Mean Reciprocal Rank) | Average rank of first correct URL | 0.8+ |
| **Precision@5** | Fraction of top-5 results that are relevant | 0.6+ |
| **Recall@5** | Fraction of relevant docs found in top-5 | 0.7+ |
| **Hit Rate@5** | Did top-5 contain any relevant doc? | 0.85+ |

**MRR Interpretation**:
- 1.0: Perfect (correct URL always rank 1)
- 0.8-0.9: Excellent (usually top 1-2)
- 0.6-0.8: Good (usually top 2-3)
- <0.4: Needs improvement

#### Answer Quality

| Metric | Description | Measurement |
|--------|-------------|-------------|
| **Factual Correctness** | Answer accuracy vs gold answer | LLM-as-judge (0-5 scale) |
| **Relevance** | Answer addresses the question | LLM evaluation |
| **Completeness** | All aspects covered | LLM evaluation |

**Output files**:
- Q&A datasets: `data/evaluation/qa_dataset_*.json`
- Results: `data/evaluation/evaluation_results_*.json`
- Reports: `data/evaluation/evaluation_report_*.html`

### Debugging Low Scores

**Low MRR**: Increase `dense_top_k`/`sparse_top_k` to 10, enable reranking

**Low Precision**: Reduce retrieval top_k, adjust reranker threshold

**Low Recall**: Increase retrieval top_k, check BM25/dense balance

---

## Project Structure

```text
hybrid-rag-system/
├── hybrid_rag_app.py          # FastAPI backend
├── ui_app.py                  # Streamlit frontend
├── test_response.py           # Local API testing
├── run_evaluation.py          # Evaluation CLI
├── src/
│   ├── config/
│   │   ├── app_config.py     # System configuration
│   │   └── evaluation_config.py
│   ├── core/                 # Retrieval components
│   ├── evaluation/           # Q&A generation + metrics
│   ├── service/              # Business logic
│   └── wiki/                 # Wikipedia scraping
└── data/
    ├── chroma_db/            # Vector database
    ├── bm25_index/           # BM25 index
    ├── evaluation/           # Q&A datasets + results
    └── corpus/               # Wikipedia articles
```

## Team

**Group 56 - WILP CAI**

| Name | Student ID |
|------|------------|
| Abhishek Kumar Tiwari | 2024AA05192 |
| Krishanu Chakraborty | 2024AA05193 |
| Viswanadha Pavan Kumar | 2024AA05197 |
| B Vinod Kumar | 2024AA05832 |
| K Abhinav | 2024AB05168 |
