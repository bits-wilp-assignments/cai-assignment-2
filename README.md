# Hybrid RAG System

A Retrieval-Augmented Generation system combining dense vector search, sparse keyword matching, and re-ranking to answer questions from Wikipedia articles.

## Quick Start

### Installation

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Application

```bash
# Terminal 1: Start backend
python hybrid_rag_app.py

# Terminal 2: Start frontend
streamlit run ui_app.py
```

Open browser at `http://localhost:8501`

### First Time Setup

1. Go to Settings page
2. Check both "Refresh Fixed" and "Refresh Random" options
3. Click "Start Indexing" (takes 10-20 minutes)
4. Wait for indexing to complete
5. Go to Chat page and ask questions

## Configuration

Edit `src/config/app_config.py` to customize:

### Key Parameters

```python
# Models
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# LLM Settings
LLM_CONFIG = {
    "max_new_tokens": 500,      # Response length
    "temperature": 0.001,        # Randomness (0.0-1.0)
    "do_sample": True,
}

# Retrieval Settings
RETRIEVAL_CONFIG = {
    "dense_top_k": 5,           # Dense retrieval results
    "sparse_top_k": 5,          # Sparse retrieval results
    "reranker_top_k": 2,        # Final documents for context
}

# Data Collection
DATA_COLLECTION_CONFIG = {
    "fixed_sample_size": 200,   # Fixed Wikipedia pages
    "random_sample_size": 300,  # Random Wikipedia pages
}
```

## API Endpoints

```bash
GET  /health              # Health check
POST /inference           # Ask questions (streaming)
POST /index/trigger       # Start indexing
GET  /index/status        # Check indexing status
GET  /config              # Get configuration
GET  /wiki/fixed          # List fixed pages
GET  /wiki/random         # List random pages
```

## Troubleshooting

### Port Already in Use
```bash
# Kill process on port 8000
lsof -i :8000  # Find PID
kill -9 <PID>  # Kill process

# Or use different port
uvicorn hybrid_rag_app:app --port 8001
```

### Out of Memory
- Use smaller models in config
- Reduce `max_new_tokens` to 200-300
- Reduce `reranker_top_k` to 1
- Set `IS_RERANKING_ENABLED = False`

### Slow Responses
- Reduce `reranker_top_k` to 1-2
- Reduce `max_new_tokens` to 200-300
- Use GPU if available

### Poor Answer Quality
- Increase `dense_top_k` and `sparse_top_k` to 10
- Increase `reranker_top_k` to 3-4
- Adjust `temperature` to 0.1-0.3

## Evaluation System

This project includes a comprehensive evaluation framework to assess RAG system performance using automatically generated Q&A pairs and industry-standard metrics.

### Quick Evaluation

```bash
# Generate 100 Q&A pairs and evaluate
python run_evaluation.py full --total-questions 100

# Generate Q&A dataset only
python run_evaluation.py generate --total-questions 100

# Evaluate with existing dataset
python run_evaluation.py evaluate --dataset-file qa_dataset_*.json

# List available datasets
python run_evaluation.py list
```

### Key Metrics

- **Mean Reciprocal Rank (MRR)**: Measures how quickly the system finds the correct source URL
- **Precision@K**: Fraction of retrieved documents that are relevant
- **Recall@K**: Fraction of relevant documents that are retrieved
- **Hit Rate@K**: Whether at least one relevant document is in top K

### Documentation

- **[EVALUATION_README.md](EVALUATION_README.md)**: Comprehensive guide to the evaluation system
- **[EVALUATION_QUICKSTART.md](EVALUATION_QUICKSTART.md)**: Quick reference and common commands
- **[example_evaluation.py](example_evaluation.py)**: Python API usage examples

### Question Types

The evaluation system generates four types of questions:

1. **Factual**: Direct questions (What, When, Where, Who)
2. **Comparative**: Comparison questions (differences, similarities)
3. **Inferential**: Reasoning questions (Why, How, implications)
4. **Multi-hop**: Complex questions requiring multiple sources

For more details, see [EVALUATION_README.md](EVALUATION_README.md)

## Project Structure

```text
hybrid-rag-system/
├── hybrid_rag_app.py          # FastAPI backend
├── ui_app.py                  # Streamlit frontend
├── run_evaluation.py          # Evaluation CLI
├── example_evaluation.py      # Example usage
├── requirements.txt           # Dependencies
├── EVALUATION_README.md       # Evaluation guide
├── EVALUATION_QUICKSTART.md   # Quick reference
├── data/                      # Data storage
│   ├── chroma_db/            # Vector database
│   ├── bm25_index/           # BM25 index
│   ├── evaluation/           # Q&A datasets and results
│   ├── fixed_wiki_pages.json
│   └── random_wiki_pages.json
├── src/
│   ├── config/               # Configuration
│   ├── core/                 # Core components
│   ├── evaluation/           # Evaluation framework
│   │   ├── qa_generator.py  # Q&A generation
│   │   ├── evaluator.py     # Metrics calculation
│   │   └── qa_storage.py    # Storage & validation
│   ├── service/              # Business logic
│   ├── util/                 # Utilities
│   └── wiki/                 # Wikipedia scraping
└── streamlit_ui/             # UI components
```

## Requirements

- Python 3.10+
- 8GB RAM (16GB recommended)
- 10GB disk space

## Team

Group 56 - WILP CAI

- Abhishek Kumar Tiwari (2024AA05192)
- Krishanu Chakraborty (2024AA05193)
- Viswanadha Pavan Kumar (2024AA05197)
- B Vinod Kumar (2024AA05832)
- K Abhinav (2024AB05168)
