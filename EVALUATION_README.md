# RAG System Evaluation Framework

Comprehensive evaluation system for assessing Hybrid RAG system performance using generated Q&A datasets.

## 📋 Overview

This evaluation framework generates diverse question-answer pairs from your Wikipedia corpus and evaluates the RAG system's retrieval performance using industry-standard metrics, with a focus on **Mean Reciprocal Rank (MRR) at the URL level**.

## 🎯 Features

- **Automatic Q&A Generation**: LLM-powered generation of 100+ diverse question types
- **Four Question Types**:
  - **Factual**: Direct questions (What, When, Where, Who)
  - **Comparative**: Comparison questions (differences, similarities)
  - **Inferential**: Reasoning questions (Why, How, implications)
  - **Multi-hop**: Complex questions requiring multiple sources
- **Comprehensive Metrics**:
  - **Mean Reciprocal Rank (MRR)** at URL level
  - **Precision@K** (K=3,5,10)
  - **Recall@K** (K=3,5,10)
  - **Hit Rate@K** (K=3,5,10)
- **Storage & Validation**: JSON-based storage with schema validation
- **Flexible Evaluation**: Evaluate entire datasets or individual questions

## 📁 Structure

```
src/evaluation/
├── qa_generator.py     # Q&A pair generation using LLM
├── evaluator.py        # Evaluation metrics (MRR, Precision, Recall)
└── qa_storage.py       # Dataset storage and validation

data/evaluation/
├── qa_dataset_*.json           # Generated Q&A datasets
└── evaluation_results_*.json   # Evaluation results

run_evaluation.py       # Main CLI script
example_evaluation.py   # Example usage scripts
```

## 🚀 Quick Start

### 1. Generate Q&A Dataset

Generate 100 Q&A pairs from your Wikipedia corpus:

```bash
python run_evaluation.py generate \
    --corpus-path ./data/fixed_wiki_pages.json \
    --total-questions 100 \
    --equal-distribution
```

**Options:**
- `--corpus-path`: Path to Wikipedia corpus JSON file
- `--total-questions`: Number of Q&A pairs to generate (default: 100)
- `--equal-distribution`: Distribute equally across all question types
- `--model-name`: LLM model for generation (default: from config)
- `--output-file`: Custom output filename

### 2. Evaluate RAG System

Evaluate your RAG system using the generated Q&A dataset:

```bash
python run_evaluation.py evaluate \
    --dataset-file qa_dataset_20260208_143022.json \
    --save-individual-results
```

**Options:**
- `--dataset-file`: Q&A dataset file to use (required)
- `--max-questions`: Limit evaluation to N questions (for testing)
- `--include-answer-generation`: Generate answers using RAG (slower)
- `--save-individual-results`: Save per-question results
- `--output-file`: Custom output filename

### 3. Full Pipeline (Generate + Evaluate)

Run the complete pipeline in one command:

```bash
python run_evaluation.py full \
    --total-questions 100 \
    --equal-distribution
```

### 4. List Available Datasets

View all generated datasets and evaluation results:

```bash
python run_evaluation.py list --type both
```

## 📊 Understanding MRR (Mean Reciprocal Rank)

**MRR at URL Level** measures how quickly your system identifies the correct source document:

- For each question, find the **rank position** of the first correct Wikipedia URL
- **Reciprocal Rank (RR)** = 1 / rank
- **MRR** = average of all RR scores

**Example:**
- Question: "What is machine learning?"
- Ground truth URL: `https://en.wikipedia.org/wiki/Machine_learning`
- If found at rank 1: RR = 1/1 = 1.0
- If found at rank 3: RR = 1/3 = 0.333
- If not found: RR = 0.0

**Interpretation:**
- MRR = 1.0: Perfect (always rank 1)
- MRR = 0.5: Average rank of 2
- MRR = 0.33: Average rank of 3
- Higher MRR = Better retrieval performance

## 📈 Evaluation Metrics

### Retrieval Metrics

1. **Mean Reciprocal Rank (MRR)**
   - Primary metric for URL-level evaluation
   - Measures speed of finding correct source

2. **Precision@K**
   - What fraction of retrieved docs are relevant?
   - Precision@5 = (Relevant docs in top 5) / 5

3. **Recall@K**
   - What fraction of relevant docs were retrieved?
   - Recall@5 = (Relevant docs in top 5) / (Total relevant docs)

4. **Hit Rate@K**
   - Is at least one relevant doc in top K?
   - Binary: 1 if yes, 0 if no

### Question Type Analysis

Results are broken down by question type:
- Factual questions performance
- Comparative questions performance
- Inferential questions performance
- Multi-hop questions performance

## 💡 Example Usage

### Python API

```python
from src.evaluation.qa_generator import QAGenerator
from src.evaluation.evaluator import RAGEvaluator
from src.evaluation.qa_storage import QAStorage

# Generate Q&A dataset
generator = QAGenerator(corpus_path="./data/fixed_wiki_pages.json")
qa_dataset = generator.generate_dataset(
    total_questions=100,
    distribution={'factual': 30, 'comparative': 25, 'inferential': 25, 'multi-hop': 20}
)

# Save dataset
storage = QAStorage()
storage.save_dataset(qa_dataset, filename="my_qa_dataset.json")

# Evaluate
evaluator = RAGEvaluator()
results = evaluator.evaluate_dataset(qa_dataset)

# Print summary
print(f"Overall MRR: {results['summary']['overall_mrr']:.4f}")
```

### Evaluate Single Question

```python
from src.evaluation.evaluator import RAGEvaluator

evaluator = RAGEvaluator()

qa_pair = {
    'question': 'What is artificial intelligence?',
    'source_urls': ['https://en.wikipedia.org/wiki/Artificial_intelligence'],
    'question_type': 'factual'
}

result = evaluator.evaluate_single_qa(qa_pair)
print(f"MRR: {result['reciprocal_rank']:.4f}")
print(f"Rank: {result['rank_of_first_correct']}")
```

## 📝 Q&A Dataset Schema

```json
{
  "question_id": "Q001",
  "question": "What is machine learning?",
  "answer": "Machine learning is a subset of AI...",
  "question_type": "factual",
  "difficulty": "medium",
  "source_ids": [12345],
  "source_urls": ["https://en.wikipedia.org/wiki/Machine_learning"],
  "source_titles": ["Machine learning"],
  "category": "Technology",
  "metadata": {
    "requires_multiple_docs": false,
    "topics": ["AI", "Technology"]
  }
}
```

## 🔧 Configuration

The evaluation system uses your existing RAG configuration from `src/config/app_config.py`:

- LLM model for Q&A generation
- Retrieval parameters (top_k, reranking, etc.)
- Embedding models

## 📊 Sample Output

```
================================================================================
RAG SYSTEM EVALUATION RESULTS
================================================================================

Total Questions Evaluated: 100

OVERALL METRICS
  Mean Reciprocal Rank (MRR):     0.7825

  Precision@3:  0.6333
  Precision@5:  0.5600
  Precision@10: 0.4250

  Recall@3:  0.7100
  Recall@5:  0.8200
  Recall@10: 0.9150

  Hit Rate@3:  0.8500
  Hit Rate@5:  0.9200
  Hit Rate@10: 0.9600

METRICS BY QUESTION TYPE

  FACTUAL (30 questions):
    MRR:          0.8567
    Precision@5:  0.6200
    Recall@5:     0.8733
    Hit Rate@5:   0.9333

  COMPARATIVE (25 questions):
    MRR:          0.7520
    Precision@5:  0.5440
    Recall@5:     0.8000
    Hit Rate@5:   0.9200

  INFERENTIAL (25 questions):
    MRR:          0.7440
    Precision@5:  0.5280
    Recall@5:     0.7920
    Hit Rate@5:   0.9000

  MULTI-HOP (20 questions):
    MRR:          0.6950
    Precision@5:  0.5400
    Recall@5:     0.8100
    Hit Rate@5:   0.9000

================================================================================
```

## 🎓 Best Practices

1. **Start Small**: Test with 20 questions first to verify setup
2. **Diverse Corpus**: Ensure your Wikipedia corpus covers multiple topics
3. **Equal Distribution**: Use `--equal-distribution` for balanced evaluation
4. **Validate Results**: Manually review a sample of generated Q&A pairs
5. **Iterative Testing**: Run evaluation multiple times and compare

## 🐛 Troubleshooting

**Q: Generation is slow**
- LLM generation takes time; expect ~5-10 seconds per question
- Use a smaller model or reduce `total_questions` for testing

**Q: Low MRR scores**
- Check if source URLs in corpus match retrieved document metadata
- Verify retrieval parameters in `app_config.py`
- Review individual results to identify failure patterns

**Q: Validation errors**
- Ensure corpus file has required fields: `page_id`, `url`, `page_title`
- Check Q&A dataset schema matches expected format

## 📚 References

- **MRR**: Measures search engine ranking quality
- **Precision/Recall**: Standard information retrieval metrics
- **RAG Evaluation**: Best practices for retrieval-augmented generation

## 🤝 Integration

This evaluation system integrates seamlessly with your existing RAG pipeline:
- Uses your configured LLM for generation
- Uses your hybrid retriever for evaluation
- Stores results alongside your data

---

**Ready to evaluate your RAG system?** Start with:

```bash
python run_evaluation.py full --total-questions 100 --equal-distribution
```
