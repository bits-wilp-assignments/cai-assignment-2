"""
Example script demonstrating how to use the evaluation system
"""
from src.evaluation.qa_generator import QAGenerator
from src.evaluation.evaluator import RAGEvaluator
from src.evaluation.qa_storage import QAStorage


def example_generate_qa():
    """Example: Generate Q&A dataset."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Generating Q&A Dataset")
    print("="*60 + "\n")
    
    # Initialize generator
    generator = QAGenerator(
        corpus_path="./data/fixed_wiki_pages.json"
    )
    
    # Generate 20 questions for quick testing
    qa_dataset = generator.generate_dataset(
        total_questions=20,
        distribution={
            'factual': 5,
            'comparative': 5,
            'inferential': 5,
            'multi-hop': 5
        }
    )
    
    # Save dataset
    storage = QAStorage()
    output_file = storage.save_dataset(qa_dataset, filename="test_qa_dataset.json")
    
    print(f"\n✓ Generated {len(qa_dataset)} Q&A pairs")
    print(f"✓ Saved to: {output_file}")
    
    # Show sample questions
    print("\nSample Questions:")
    for i, qa in enumerate(qa_dataset[:3], 1):
        print(f"\n{i}. [{qa['question_type']}] {qa['question']}")
        print(f"   Answer: {qa['answer'][:100]}...")


def example_evaluate_rag():
    """Example: Evaluate RAG system."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Evaluating RAG System")
    print("="*60 + "\n")
    
    # Load Q&A dataset
    storage = QAStorage()
    datasets = storage.list_datasets()
    
    if not datasets:
        print("⚠ No datasets found. Generate one first using example_generate_qa()")
        return
    
    # Use the most recent dataset
    dataset_file = datasets[-1]
    print(f"Using dataset: {dataset_file}")
    
    qa_dataset = storage.load_dataset(dataset_file)
    
    # Evaluate (limit to 5 questions for quick demo)
    evaluator = RAGEvaluator()
    results = evaluator.evaluate_dataset(
        qa_dataset[:5],
        include_answer_generation=False
    )
    
    # Print results
    summary = results['summary']
    print(f"\n✓ Evaluated {summary['total_questions']} questions")
    print(f"✓ Overall MRR: {summary['overall_mrr']:.4f}")
    print(f"✓ Precision@5: {summary['overall_metrics']['precision_at_5']:.4f}")
    print(f"✓ Recall@5: {summary['overall_metrics']['recall_at_5']:.4f}")
    
    # Save results
    output_file = storage.save_evaluation_results(results, filename="test_evaluation_results.json")
    print(f"✓ Results saved to: {output_file}")


def example_single_question():
    """Example: Evaluate a single question."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Evaluating Single Question")
    print("="*60 + "\n")
    
    # Create a test Q&A pair
    test_qa = {
        'question_id': 'TEST001',
        'question': 'What is artificial intelligence?',
        'answer': 'AI is the simulation of human intelligence by machines.',
        'question_type': 'factual',
        'source_ids': [12345],
        'source_urls': ['https://en.wikipedia.org/wiki/Artificial_intelligence'],
        'difficulty': 'easy'
    }
    
    # Evaluate
    evaluator = RAGEvaluator()
    result = evaluator.evaluate_single_qa(test_qa)
    
    # Print results
    print(f"Question: {test_qa['question']}")
    print(f"\nReciprocal Rank: {result['reciprocal_rank']:.4f}")
    print(f"Rank of First Correct URL: {result['rank_of_first_correct']}")
    print(f"Precision@5: {result['precision_at_5']:.4f}")
    print(f"Hit Rate@10: {result['hit_rate_at_10']:.4f}")
    
    print("\nRetrieved URLs (top 3):")
    for i, url in enumerate(result['retrieved_urls'][:3], 1):
        print(f"  {i}. {url}")


def main():
    """Run examples."""
    print("\n" + "="*60)
    print("RAG EVALUATION SYSTEM - EXAMPLES")
    print("="*60)
    
    # Uncomment the examples you want to run:
    
    # Example 1: Generate Q&A dataset
    # example_generate_qa()
    
    # Example 2: Evaluate RAG system
    # example_evaluate_rag()
    
    # Example 3: Evaluate single question
    example_single_question()
    
    print("\n" + "="*60)
    print("Examples complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
