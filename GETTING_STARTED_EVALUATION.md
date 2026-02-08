# Getting Started with RAG Evaluation - Step by Step

## Prerequisites Check

Before starting, ensure you have:
- ✅ Python 3.10+ installed
- ✅ Virtual environment activated
- ✅ Requirements installed (`pip install -r requirements.txt`)
- ✅ RAG system indexed (corpus files exist in `data/`)

## Step 1: Verify Installation

Run the test script to ensure everything is set up correctly:

```bash
python test_evaluation_setup.py
```

**Expected output:**
```
==============================================================
EVALUATION SYSTEM - INSTALLATION TEST
==============================================================

Testing imports...
✓ Evaluation modules imported successfully

Checking corpus file...
✓ Found corpus: ./data/fixed_wiki_pages.json

Checking storage directory...
✓ Storage directory ready: ./data/evaluation

Testing basic functionality...
✓ Q&A validation working

==============================================================
TEST SUMMARY
==============================================================
✓ PASS   Module Imports
✓ PASS   Corpus File
✓ PASS   Storage Directory
✓ PASS   Basic Functionality

Passed: 4/4

✓ All tests passed! Evaluation system is ready.
```

If any tests fail, follow the troubleshooting hints in the output.

## Step 2: Generate Your First Q&A Dataset

Start with a small dataset (20 questions) to test the system:

```bash
python run_evaluation.py generate \
    --total-questions 20 \
    --equal-distribution \
    --output-file test_qa_20.json
```

**What happens:**
1. Script loads your Wikipedia corpus
2. Samples diverse documents across categories
3. Uses LLM to generate questions and answers
4. Validates and saves to `data/evaluation/test_qa_20.json`

**Expected time:** 3-5 minutes (depending on your machine and LLM)

**Output:**
```
============================================================
Q&A GENERATION SUMMARY
============================================================
Total Questions: 20

Distribution:
  factual        : 5/5
  comparative    : 5/5
  inferential    : 5/5
  multi-hop      : 5/5

Output File: ./data/evaluation/test_qa_20.json
============================================================
```

## Step 3: Review Generated Questions

Before evaluating, manually review some generated questions:

```bash
# View the dataset
cat data/evaluation/test_qa_20.json | head -n 50
```

Or use Python:

```python
import json

with open('data/evaluation/test_qa_20.json', 'r') as f:
    data = json.load(f)

# Show first 3 questions
for qa in data['questions'][:3]:
    print(f"\nQ: {qa['question']}")
    print(f"A: {qa['answer']}")
    print(f"Type: {qa['question_type']}")
    print(f"Source: {qa['source_urls'][0]}")
```

**Quality check:**
- Are questions clear and answerable?
- Are answers accurate based on the source?
- Are question types diverse?

If quality is poor, you may need to adjust LLM parameters or regenerate.

## Step 4: Run Your First Evaluation

Evaluate your RAG system with the test dataset:

```bash
python run_evaluation.py evaluate \
    --dataset-file test_qa_20.json \
    --save-individual-results
```

**What happens:**
1. Loads the Q&A dataset
2. For each question:
   - Retrieves documents using your hybrid retriever
   - Calculates MRR (finds rank of correct URL)
   - Calculates Precision, Recall, Hit Rate
3. Aggregates metrics by question type
4. Saves results to `data/evaluation/evaluation_results_*.json`

**Expected time:** 2-4 minutes for 20 questions

**Output:**
```
================================================================================
RAG SYSTEM EVALUATION RESULTS
================================================================================

Total Questions Evaluated: 20

OVERALL METRICS
  Mean Reciprocal Rank (MRR):     0.7250

  Precision@3:  0.5667
  Precision@5:  0.4800
  Precision@10: 0.3650

  Recall@3:  0.6500
  Recall@5:  0.7800
  Recall@10: 0.8900

  Hit Rate@3:  0.8000
  Hit Rate@5:  0.9000
  Hit Rate@10: 0.9500

METRICS BY QUESTION TYPE

  FACTUAL (5 questions):
    MRR:          0.8000
    Precision@5:  0.5200
    Recall@5:     0.8200
    Hit Rate@5:   1.0000

  COMPARATIVE (5 questions):
    MRR:          0.7000
    Precision@5:  0.4800
    Recall@5:     0.7600
    Hit Rate@5:   0.8000

  ... (and so on)
================================================================================
```

## Step 5: Interpret Results

### Understanding Your MRR Score

Your MRR tells you how quickly your system finds the correct source:

| MRR Range | Quality | Typical Rank | Action |
|-----------|---------|-------------|---------|
| 0.8 - 1.0 | Excellent | 1-2 | System working well |
| 0.6 - 0.8 | Good | 2-3 | Minor tuning may help |
| 0.4 - 0.6 | Fair | 3-5 | Review retrieval config |
| < 0.4 | Poor | 5+ | Major issues to address |

### Analyzing Question Types

Look at per-type metrics to identify weaknesses:

**Example findings:**
- Factual MRR = 0.80 → Good at simple questions
- Multi-hop MRR = 0.50 → Struggles with complex queries
- **Action:** Increase `rrf_top_k` or enable reranking

### Other Metrics

- **High Recall, Low Precision**: Retrieving too many irrelevant docs → Reduce top_k
- **Low Recall, High Precision**: Missing relevant docs → Increase top_k
- **Low Hit Rate**: Not finding sources at all → Check indexing or query processing

## Step 6: Generate Full Dataset

Once you're satisfied with the test, generate the full 100-question dataset:

```bash
python run_evaluation.py full \
    --total-questions 100 \
    --equal-distribution
```

This runs both generation and evaluation in one command.

**Expected time:** 20-30 minutes total

## Step 7: Track Improvements

After making changes to your RAG system:

1. **Re-run evaluation** with the same dataset:
   ```bash
   python run_evaluation.py evaluate --dataset-file qa_dataset_*.json
   ```

2. **Compare MRR scores** before and after:
   ```
   Before: MRR = 0.65
   After:  MRR = 0.78  (+20% improvement!)
   ```

3. **Keep a log** of configurations and results:
   ```
   Version 1: MRR = 0.65 (baseline)
   Version 2: MRR = 0.72 (enabled reranking)
   Version 3: MRR = 0.78 (increased top_k to 10)
   ```

## Common First-Time Issues

### Issue 1: "Out of memory"
**Solution:** Use smaller LLM or reduce `max_new_tokens`:
```python
# In src/config/app_config.py
LLM_CONFIG = {
    "max_new_tokens": 200,  # Reduce from 500
    ...
}
```

### Issue 2: "Generation is very slow"
**Normal:** LLM generation takes time (~5-10 seconds per question)
**Workaround:** Start with 10-20 questions for testing

### Issue 3: "Low quality questions"
**Solution:** 
- Review and filter bad questions manually
- Adjust LLM temperature (try 0.5-0.7)
- Try different model if available

### Issue 4: "Can't find corpus file"
**Solution:** Make sure you're in project root and have run indexing:
```bash
# Check current directory
pwd  # Should be: .../hybrid-rag-system

# Check corpus exists
ls data/*.json

# If missing, run indexing first
python hybrid_rag_app.py  # Start backend
# Then use UI to trigger indexing
```

### Issue 5: "MRR is 0.0 or very low"
**Possible causes:**
1. URL field mismatch between corpus and retrieved docs
2. Retrieval system not working properly
3. Ground truth URLs incorrect

**Debug:**
```python
# Check what URLs are being retrieved
python example_evaluation.py  # Run example_single_question()
# Look at retrieved_urls in output
```

## Next Steps After First Evaluation

1. **Analyze Results**: Review per-question-type metrics
2. **Identify Bottlenecks**: Which question types perform worst?
3. **Tune Parameters**: Adjust retrieval config in `app_config.py`
4. **Re-evaluate**: Run evaluation again with same dataset
5. **Document Progress**: Keep notes on what changes helped
6. **Scale Up**: Once confident, use larger datasets

## Quick Reference Commands

```bash
# Test installation
python test_evaluation_setup.py

# Generate test dataset (20 questions)
python run_evaluation.py generate --total-questions 20

# Evaluate with test dataset
python run_evaluation.py evaluate --dataset-file test_qa_20.json

# Full pipeline (100 questions)
python run_evaluation.py full --total-questions 100

# List all datasets and results
python run_evaluation.py list

# Run Python examples
python example_evaluation.py
```

## Getting Help

- **Full Documentation**: [EVALUATION_README.md](EVALUATION_README.md)
- **Quick Reference**: [EVALUATION_QUICKSTART.md](EVALUATION_QUICKSTART.md)
- **Implementation Details**: [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
- **Main README**: [README.md](README.md)

## Success Criteria

You're ready to move on when:
- ✅ Test script passes all checks
- ✅ Can generate 20 Q&A pairs successfully
- ✅ Can run evaluation and get results
- ✅ Understand MRR and other metrics
- ✅ Can interpret results to identify issues

**Congratulations! You now have a working RAG evaluation system.**

Continue to [EVALUATION_README.md](EVALUATION_README.md) for advanced usage and optimization strategies.
