# RAG Evaluation - Quick Reference Guide

## 🚀 Common Commands

### Generate 100 Q&A Pairs (Equal Distribution)
```bash
python run_evaluation.py generate \
    --total-questions 100 \
    --equal-distribution
```

### Generate with Custom Distribution
```bash
python run_evaluation.py generate \
    --total-questions 100
    # Default: 30 factual, 25 comparative, 25 inferential, 20 multi-hop
```

### Evaluate RAG System
```bash
python run_evaluation.py evaluate \
    --dataset-file qa_dataset_20260208_143022.json
```

### Full Pipeline (Generate + Evaluate)
```bash
python run_evaluation.py full --total-questions 100
```

### Quick Test (20 questions)
```bash
python run_evaluation.py full \
    --total-questions 20 \
    --max-eval-questions 10
```

## 📊 Interpreting Results

### MRR (Mean Reciprocal Rank)
- **1.0**: Perfect - always finds correct URL at rank 1
- **0.8-0.9**: Excellent - usually top 1-2
- **0.6-0.8**: Good - usually top 2-3
- **0.4-0.6**: Fair - usually top 3-5
- **<0.4**: Needs improvement

### Precision@5
- Percentage of top 5 results that are relevant
- **0.8+**: Excellent
- **0.6-0.8**: Good
- **0.4-0.6**: Fair
- **<0.4**: Needs improvement

### Recall@5
- Percentage of relevant docs found in top 5
- **0.9+**: Excellent coverage
- **0.7-0.9**: Good coverage
- **0.5-0.7**: Fair coverage
- **<0.5**: Poor coverage

### Hit Rate@5
- Did we find at least one relevant doc in top 5?
- **0.95+**: Excellent
- **0.85-0.95**: Good
- **0.70-0.85**: Fair
- **<0.70**: Needs improvement

## 🎯 Workflow

1. **Generate Q&A Dataset** (once)
   ```bash
   python run_evaluation.py generate --total-questions 100
   ```

2. **Run Evaluation** (after changes)
   ```bash
   python run_evaluation.py evaluate --dataset-file <filename>
   ```

3. **Compare Results**
   - Adjust retrieval parameters in `app_config.py`
   - Re-run evaluation
   - Compare MRR and other metrics

4. **Iterate**
   - Identify weak question types
   - Tune system accordingly
   - Re-evaluate

## 🔍 Debugging Low Scores

### If MRR is low:
1. Check URL matching in retrieved documents
2. Verify retrieval top_k is sufficient
3. Review individual failed questions
4. Check if reranking is enabled

### If Precision is low:
1. Too many irrelevant docs retrieved
2. Consider stricter filtering
3. Adjust reranker threshold

### If Recall is low:
1. Increase retrieval top_k
2. Check BM25 and dense retrieval balance
3. Verify document chunking strategy

## 📁 File Locations

- **Q&A Datasets**: `data/evaluation/qa_dataset_*.json`
- **Evaluation Results**: `data/evaluation/evaluation_results_*.json`
- **Corpus**: `data/fixed_wiki_pages.json`
- **Configuration**: `src/config/app_config.py`

## 🛠️ Customization

### Change LLM for Generation
Edit `src/config/app_config.py`:
```python
LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"  # Use larger model
```

### Adjust Retrieval Parameters
Edit `src/config/app_config.py`:
```python
RETRIEVAL_CONFIG = {
    "dense_top_k": 10,     # Increase for more coverage
    "sparse_top_k": 10,
    "rrf_top_k": 8,
    "reranker_top_k": 5,
    "rrf_k": 60,
}
```

### Custom Question Distribution
In Python:
```python
from src.evaluation import QAGenerator

generator = QAGenerator("./data/fixed_wiki_pages.json")
qa_dataset = generator.generate_dataset(
    total_questions=100,
    distribution={
        'factual': 40,      # More factual
        'comparative': 20,
        'inferential': 20,
        'multi-hop': 20
    }
)
```

## 💡 Tips

1. **Start with 20 questions** to test the system
2. **Manual review** 5-10 generated Q&A pairs for quality
3. **Run baseline evaluation** before making changes
4. **Track MRR over time** as you improve the system
5. **Focus on question types** where performance is lowest

## 🐛 Common Issues

**Issue**: "No module named 'src.evaluation'"
```bash
# Run from project root directory
cd /path/to/hybrid-rag-system
python run_evaluation.py ...
```

**Issue**: "Dataset file not found"
```bash
# Check the data/evaluation/ directory for available datasets
ls data/evaluation/qa_dataset_*.json
# Use exact filename
python run_evaluation.py evaluate --dataset-file qa_dataset_20260208_143022.json
```

**Issue**: "Out of memory during generation"
```python
# Use smaller model or reduce max_new_tokens
# Edit src/config/app_config.py
LLM_CONFIG = {
    "max_new_tokens": 200,  # Reduce from 500
    ...
}
```

## 📞 Need Help?

Refer to [EVALUATION_README.md](EVALUATION_README.md) for comprehensive documentation.
