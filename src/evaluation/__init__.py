"""
RAG Evaluation Module
=====================

This module provides tools for evaluating RAG system performance through
automated Q&A generation and comprehensive metrics analysis.

Components:
-----------
- QAGenerator: Generates diverse Q&A pairs from Wikipedia corpus
- RAGEvaluator: Evaluates retrieval performance with MRR and other metrics
- QAStorage: Handles dataset storage, validation, and management

Quick Start:
-----------
    from src.evaluation import QAGenerator, RAGEvaluator, QAStorage
    
    # Generate Q&A dataset
    generator = QAGenerator("./data/fixed_wiki_pages.json")
    qa_dataset = generator.generate_dataset(total_questions=100)
    
    # Evaluate RAG system
    evaluator = RAGEvaluator()
    results = evaluator.evaluate_dataset(qa_dataset)
    
    # Save results
    storage = QAStorage()
    storage.save_evaluation_results(results)

For detailed documentation, see EVALUATION_README.md
"""

from src.evaluation.qa_generator import QAGenerator
from src.evaluation.evaluator import RAGEvaluator
from src.evaluation.qa_storage import QAStorage

__all__ = ['QAGenerator', 'RAGEvaluator', 'QAStorage']
__version__ = '1.0.0'
