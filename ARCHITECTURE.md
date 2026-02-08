# RAG Evaluation System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RAG EVALUATION SYSTEM                                │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                           INPUT: Wikipedia Corpus                            │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │  data/fixed_wiki_pages.json                                         │    │
│  │  data/random_wiki_pages.json                                        │    │
│  │                                                                      │    │
│  │  {                                                                   │    │
│  │    "page_id": 12345,                                                │    │
│  │    "page_title": "Machine Learning",                                │    │
│  │    "url": "https://en.wikipedia.org/wiki/Machine_learning",         │    │
│  │    "extract": "Machine learning is...",                             │    │
│  │    "category": "Technology"                                          │    │
│  │  }                                                                   │    │
│  └────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STEP 1: Q&A GENERATION                                    │
│                   (src/evaluation/qa_generator.py)                           │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────┐      │
│  │  QAGenerator                                                      │      │
│  │  ─────────────────────────────────────────────────────────────   │      │
│  │  • Loads Wikipedia corpus                                         │      │
│  │  • Samples documents (diverse or random)                          │      │
│  │  • Uses LLM to generate questions                                 │      │
│  │  • Generates ground truth answers                                 │      │
│  │  • Validates Q&A pairs                                            │      │
│  │                                                                    │      │
│  │  Question Types:                                                  │      │
│  │  ┌────────────┬────────────┬────────────┬────────────┐          │      │
│  │  │  Factual   │Comparative │Inferential │ Multi-hop  │          │      │
│  │  │   (25%)    │   (25%)    │   (25%)    │   (25%)    │          │      │
│  │  └────────────┴────────────┴────────────┴────────────┘          │      │
│  └──────────────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STEP 2: STORAGE & VALIDATION                              │
│                    (src/evaluation/qa_storage.py)                            │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────┐      │
│  │  QAStorage                                                        │      │
│  │  ──────────                                                       │      │
│  │  • Validates Q&A pairs against schema                            │      │
│  │  • Saves to JSON with metadata                                   │      │
│  │  • Provides load/list/merge operations                           │      │
│  │                                                                   │      │
│  │  Output: data/evaluation/qa_dataset_TIMESTAMP.json               │      │
│  │  {                                                                │      │
│  │    "total_questions": 100,                                        │      │
│  │    "metadata": { ... },                                           │      │
│  │    "questions": [                                                 │      │
│  │      {                                                            │      │
│  │        "question_id": "Q001",                                     │      │
│  │        "question": "What is machine learning?",                   │      │
│  │        "answer": "Machine learning is...",                        │      │
│  │        "question_type": "factual",                                │      │
│  │        "source_urls": ["https://..."],                            │      │
│  │        "difficulty": "medium"                                     │      │
│  │      },                                                           │      │
│  │      ...                                                          │      │
│  │    ]                                                              │      │
│  │  }                                                                │      │
│  └──────────────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     STEP 3: RAG SYSTEM EVALUATION                            │
│                     (src/evaluation/evaluator.py)                            │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────┐      │
│  │  RAGEvaluator                                                     │      │
│  │  ────────────                                                     │      │
│  │                                                                   │      │
│  │  For each Q&A pair:                                               │      │
│  │                                                                   │      │
│  │  1. Retrieve documents using HybridRetriever                     │      │
│  │     ┌─────────────────────────────────────────┐                 │      │
│  │     │  Query: "What is machine learning?"     │                 │      │
│  │     │           ↓                               │                 │      │
│  │     │  ┌──────────────────────────┐           │                 │      │
│  │     │  │   Hybrid Retriever        │           │                 │      │
│  │     │  │  • Dense (Vector Search) │           │                 │      │
│  │     │  │  • Sparse (BM25)         │           │                 │      │
│  │     │  │  • RRF Fusion            │           │                 │      │
│  │     │  │  • Reranking             │           │                 │      │
│  │     │  └──────────────────────────┘           │                 │      │
│  │     │           ↓                               │                 │      │
│  │     │  Retrieved Documents (Ranked):           │                 │      │
│  │     │  1. doc1 (url: https://...)              │                 │      │
│  │     │  2. doc2 (url: https://...)              │                 │      │
│  │     │  3. doc3 (url: https://...)              │                 │      │
│  │     │  ...                                      │                 │      │
│  │     └─────────────────────────────────────────┘                 │      │
│  │                                                                   │      │
│  │  2. Calculate MRR at URL Level                                   │      │
│  │     ┌─────────────────────────────────────────┐                 │      │
│  │     │  Ground Truth URL:                       │                 │      │
│  │     │  https://en.wikipedia.org/wiki/ML        │                 │      │
│  │     │                                           │                 │      │
│  │     │  Find in retrieved results:              │                 │      │
│  │     │  Rank 1: https://...                    │                 │      │
│  │     │  Rank 2: https://...                    │                 │      │
│  │     │  Rank 3: https://.../ML ✓ FOUND!        │                 │      │
│  │     │                                           │                 │      │
│  │     │  Reciprocal Rank = 1/3 = 0.333          │                 │      │
│  │     └─────────────────────────────────────────┘                 │      │
│  │                                                                   │      │
│  │  3. Calculate Other Metrics                                      │      │
│  │     • Precision@K: Relevance of top K                           │      │
│  │     • Recall@K: Coverage of relevant docs                       │      │
│  │     • Hit Rate@K: Found at least one?                           │      │
│  │                                                                   │      │
│  └──────────────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STEP 4: AGGREGATE & ANALYZE                               │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────┐      │
│  │  Aggregate Metrics                                                │      │
│  │  ─────────────────                                                │      │
│  │                                                                   │      │
│  │  Overall (100 questions):                                         │      │
│  │    MRR = 0.7825              ← Primary Metric                     │      │
│  │    Precision@5 = 0.56                                             │      │
│  │    Recall@5 = 0.82                                                │      │
│  │    Hit Rate@5 = 0.92                                              │      │
│  │                                                                   │      │
│  │  By Question Type:                                                │      │
│  │  ┌──────────────┬──────┬────────┬─────────┬──────────┐          │      │
│  │  │ Type         │ MRR  │ Prec@5 │ Recall@5│ HitRate@5│          │      │
│  │  ├──────────────┼──────┼────────┼─────────┼──────────┤          │      │
│  │  │ Factual      │ 0.86 │ 0.62   │ 0.87    │ 0.93     │          │      │
│  │  │ Comparative  │ 0.75 │ 0.54   │ 0.80    │ 0.92     │          │      │
│  │  │ Inferential  │ 0.74 │ 0.53   │ 0.79    │ 0.90     │          │      │
│  │  │ Multi-hop    │ 0.70 │ 0.54   │ 0.81    │ 0.90     │          │      │
│  │  └──────────────┴──────┴────────┴─────────┴──────────┘          │      │
│  │                                                                   │      │
│  │  Individual Question Results: Available for deep analysis         │      │
│  │                                                                   │      │
│  └──────────────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STEP 5: SAVE RESULTS                                      │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────┐      │
│  │  data/evaluation/evaluation_results_TIMESTAMP.json               │      │
│  │  {                                                                │      │
│  │    "evaluated_at": "2026-02-08T14:30:22",                        │      │
│  │    "summary": {                                                   │      │
│  │      "total_questions": 100,                                      │      │
│  │      "overall_mrr": 0.7825,                                       │      │
│  │      "overall_metrics": { ... },                                  │      │
│  │      "metrics_by_question_type": { ... }                          │      │
│  │    },                                                             │      │
│  │    "individual_results": [ ... ]                                  │      │
│  │  }                                                                │      │
│  └──────────────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACES                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. Command Line Interface (run_evaluation.py)                              │
│     ┌────────────────────────────────────────────────────────────┐         │
│     │  python run_evaluation.py generate --total-questions 100   │         │
│     │  python run_evaluation.py evaluate --dataset-file <file>   │         │
│     │  python run_evaluation.py full --total-questions 100       │         │
│     │  python run_evaluation.py list                             │         │
│     └────────────────────────────────────────────────────────────┘         │
│                                                                              │
│  2. Python API (src/evaluation/__init__.py)                                 │
│     ┌────────────────────────────────────────────────────────────┐         │
│     │  from src.evaluation import (                              │         │
│     │      QAGenerator, RAGEvaluator, QAStorage                  │         │
│     │  )                                                          │         │
│     │                                                             │         │
│     │  generator = QAGenerator(corpus_path)                      │         │
│     │  qa_dataset = generator.generate_dataset(100)              │         │
│     │                                                             │         │
│     │  evaluator = RAGEvaluator()                                │         │
│     │  results = evaluator.evaluate_dataset(qa_dataset)          │         │
│     └────────────────────────────────────────────────────────────┘         │
│                                                                              │
│  3. Example Scripts (example_evaluation.py)                                 │
│     ┌────────────────────────────────────────────────────────────┐         │
│     │  python example_evaluation.py                              │         │
│     │  • Demonstrates basic usage                                │         │
│     │  • Single question evaluation                              │         │
│     │  • Quick testing examples                                  │         │
│     └────────────────────────────────────────────────────────────┘         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                         KEY METRICS EXPLAINED                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Mean Reciprocal Rank (MRR) - Primary Metric                                │
│  ────────────────────────────────────────────                               │
│  • Measures speed of finding correct source URL                             │
│  • MRR = Average(1/rank) across all questions                               │
│  • Range: 0.0 to 1.0 (higher is better)                                     │
│  • Interpretation:                                                           │
│    - 1.0: Always rank 1 (perfect)                                           │
│    - 0.5: Average rank 2                                                     │
│    - 0.33: Average rank 3                                                    │
│                                                                              │
│  Precision@K                                                                 │
│  ───────────                                                                 │
│  • Fraction of retrieved docs that are relevant                             │
│  • Precision@5 = (Relevant in top 5) / 5                                    │
│  • Measures accuracy of retrieval                                           │
│                                                                              │
│  Recall@K                                                                    │
│  ─────────                                                                   │
│  • Fraction of relevant docs that are retrieved                             │
│  • Recall@5 = (Relevant in top 5) / (Total relevant)                        │
│  • Measures coverage                                                         │
│                                                                              │
│  Hit Rate@K                                                                  │
│  ───────────                                                                 │
│  • Did we find at least one relevant doc?                                   │
│  • Binary: 1 if yes, 0 if no                                                │
│  • Measures minimum success                                                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                         INTEGRATION POINTS                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Existing RAG System Components Used:                                        │
│                                                                              │
│  1. LLMFactory (src/core/llm.py)                                            │
│     └─→ Used for Q&A generation                                             │
│                                                                              │
│  2. HybridRetriever (src/core/retriever.py)                                 │
│     └─→ Used for document retrieval during evaluation                       │
│                                                                              │
│  3. Configuration (src/config/app_config.py)                                │
│     └─→ LLM model, retrieval parameters                                     │
│                                                                              │
│  4. Wikipedia Corpus (data/*.json)                                          │
│     └─→ Source for Q&A generation                                           │
│                                                                              │
│  No modifications to existing RAG system required!                           │
│  Evaluation system is completely standalone but integrated.                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                         WORKFLOW SUMMARY                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Development Cycle:                                                          │
│                                                                              │
│  1. Generate Q&A Dataset (once)                                             │
│     └─→ python run_evaluation.py generate --total-questions 100             │
│                                                                              │
│  2. Establish Baseline                                                       │
│     └─→ python run_evaluation.py evaluate --dataset-file <file>             │
│     └─→ Note: MRR = 0.65 (baseline)                                         │
│                                                                              │
│  3. Tune RAG System                                                          │
│     └─→ Adjust parameters in app_config.py                                  │
│     └─→ Modify retrieval strategy                                           │
│                                                                              │
│  4. Re-evaluate                                                              │
│     └─→ python run_evaluation.py evaluate --dataset-file <same_file>        │
│     └─→ Note: MRR = 0.78 (improved!)                                        │
│                                                                              │
│  5. Compare & Iterate                                                        │
│     └─→ Track improvements over time                                        │
│     └─→ Focus on weak question types                                        │
│     └─→ Repeat steps 3-5                                                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## File Inventory

### Core Modules (src/evaluation/)
- `qa_generator.py` (400+ lines): Q&A pair generation with LLM
- `evaluator.py` (400+ lines): Metrics calculation (MRR, Precision, Recall)
- `qa_storage.py` (300+ lines): Storage, validation, dataset management
- `__init__.py`: Module initialization and exports

### Scripts & Runners
- `run_evaluation.py` (400+ lines): Main CLI interface
- `example_evaluation.py` (150+ lines): Usage examples
- `test_evaluation_setup.py` (120+ lines): Installation verification

### Documentation
- `EVALUATION_README.md`: Comprehensive documentation
- `EVALUATION_QUICKSTART.md`: Quick reference guide
- `GETTING_STARTED_EVALUATION.md`: Step-by-step tutorial
- `IMPLEMENTATION_SUMMARY.md`: Technical implementation details
- `ARCHITECTURE.md` (this file): Visual architecture overview

### Data Directories
- `data/evaluation/`: Q&A datasets and evaluation results
- `data/fixed_wiki_pages.json`: Source corpus
- `data/random_wiki_pages.json`: Additional corpus

**Total Implementation**: ~2,000+ lines of production code + extensive documentation
