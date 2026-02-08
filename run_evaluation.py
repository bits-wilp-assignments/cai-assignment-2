"""
RAG Evaluation Runner
Main script to generate Q&A pairs and evaluate RAG system
"""
import argparse
import os
from src.evaluation.qa_generator import QAGenerator
from src.evaluation.evaluator import RAGEvaluator
from src.evaluation.qa_storage import QAStorage
from src.evaluation.report_generator import generate_html_report
from src.util.logging_util import get_logger
from src.config.app_config import LLM_MODEL
from src.config.evaluation_config import (
    DEFAULT_CORPUS_PATH,
    DEFAULT_TOTAL_QUESTIONS,
    DEFAULT_QUESTION_DISTRIBUTION
)


def print_evaluation_summary(results: dict):
    """Pretty print evaluation summary."""
    summary = results['summary']

    print("\n" + "="*80)
    print("RAG SYSTEM EVALUATION RESULTS")
    print("="*80)

    print(f"\nTotal Questions in Dataset: {summary['total_questions']}")

    # Show actual evaluation counts if limits were applied
    retrieval_count = summary.get('questions_evaluated_retrieval', 0)
    answer_count = summary.get('questions_evaluated_answer', 0)

    if retrieval_count > 0 or answer_count > 0:
        print(f"Questions Evaluated:")
        if retrieval_count > 0:
            print(f"  - Retrieval Quality: {retrieval_count}")
        if answer_count > 0:
            print(f"  - Answer Quality: {answer_count}")

    answer_only_mode = summary.get('answer_only_mode', False)

    if answer_only_mode:
        print(f"\n{'ANSWER QUALITY METRICS (ANSWER-ONLY MODE)':-^80}")

        # Show answer quality metrics if evaluated
        if summary.get('answer_evaluation_enabled'):
            if summary.get('overall_f1') is not None:
                print(f"  F1 Score:                       {summary['overall_f1']:.4f}")
            if summary.get('overall_bleu') is not None:
                print(f"  BLEU Score:                     {summary['overall_bleu']:.4f}")
            if summary.get('overall_rouge_l') is not None:
                print(f"  ROUGE-L Score:                  {summary['overall_rouge_l']:.4f}")
            if summary.get('overall_semantic_similarity') is not None:
                print(f"  Semantic Similarity:            {summary['overall_semantic_similarity']:.4f}")
    else:
        print(f"\n{'OVERALL METRICS':-^80}")

        # Show MRR if not in answer_only mode
        if summary.get('overall_mrr') is not None:
            print(f"  Mean Reciprocal Rank (MRR):     {summary['overall_mrr']:.4f}")

        # Show answer quality metrics if evaluated
        if summary.get('answer_evaluation_enabled'):
            print(f"\n  Answer Quality Metrics:")
            if summary.get('overall_f1') is not None:
                print(f"    F1 Score:                     {summary['overall_f1']:.4f}")
            if summary.get('overall_bleu') is not None:
                print(f"    BLEU Score:                   {summary['overall_bleu']:.4f}")
            if summary.get('overall_rouge_l') is not None:
                print(f"    ROUGE-L Score:                {summary['overall_rouge_l']:.4f}")
            if summary.get('overall_semantic_similarity') is not None:
                print(f"    Semantic Similarity:          {summary['overall_semantic_similarity']:.4f}")

        if 'overall_metrics' in summary:
            overall = summary['overall_metrics']
            print(f"\n  Precision@3:  {overall['precision_at_3']:.4f}")
            print(f"  Precision@5:  {overall['precision_at_5']:.4f}")
            print(f"  Precision@10: {overall['precision_at_10']:.4f}")

            print(f"\n  Recall@3:  {overall['recall_at_3']:.4f}")
            print(f"  Recall@5:  {overall['recall_at_5']:.4f}")
            print(f"  Recall@10: {overall['recall_at_10']:.4f}")

            print(f"\n  Hit Rate@3:  {overall['hit_rate_at_3']:.4f}")
            print(f"  Hit Rate@5:  {overall['hit_rate_at_5']:.4f}")
            print(f"  Hit Rate@10: {overall['hit_rate_at_10']:.4f}")

    print(f"\n{'METRICS BY QUESTION TYPE':-^80}")

    for qtype, metrics in summary['metrics_by_question_type'].items():
        print(f"\n  {qtype.upper()} ({metrics['count']} questions):")

        if not answer_only_mode:
            if 'mrr' in metrics:
                print(f"    MRR:          {metrics['mrr']:.4f}")
            if 'precision_at_5' in metrics:
                print(f"    Precision@5:  {metrics['precision_at_5']:.4f}")
            if 'recall_at_5' in metrics:
                print(f"    Recall@5:     {metrics['recall_at_5']:.4f}")
            if 'hit_rate_at_5' in metrics:
                print(f"    Hit Rate@5:   {metrics['hit_rate_at_5']:.4f}")

        if 'answer_similarity' in metrics or 'f1_score' in metrics:
            print(f"    Answer Quality:")
            if 'f1_score' in metrics:
                print(f"      F1:           {metrics['f1_score']:.4f}")
            if 'bleu_score' in metrics:
                print(f"      BLEU:         {metrics['bleu_score']:.4f}")
            if 'rouge_l' in metrics:
                print(f"      ROUGE-L:      {metrics['rouge_l']:.4f}")
            if 'semantic_similarity' in metrics:
                print(f"      Semantic Sim: {metrics['semantic_similarity']:.4f}")
            # Legacy support
            elif 'answer_similarity' in metrics:
                print(f"      Semantic Sim: {metrics['answer_similarity']:.4f}")

    print("\n" + "="*80 + "\n")


def generate_qa_dataset(args):
    """Generate Q&A dataset."""
    logger = get_logger(__name__)
    logger.info("Starting Q&A dataset generation...")

    # Initialize generator
    generator = QAGenerator(
        corpus_path=args.corpus_path,
        model_name=args.model_name
    )

    # Define distribution
    if args.equal_distribution:
        # Equal distribution across all types
        per_type = args.total_questions // 4
        distribution = {
            'factual': per_type,
            'comparative': per_type,
            'inferential': per_type,
            'multi-hop': per_type
        }
    else:
        # Custom distribution from config
        distribution = DEFAULT_QUESTION_DISTRIBUTION.copy()

    # Generate dataset
    qa_dataset = generator.generate_dataset(
        total_questions=args.total_questions,
        distribution=distribution
    )

    # Save dataset
    storage = QAStorage()
    output_file = storage.save_dataset(
        qa_dataset,
        filename=args.output_file,
        validate=True
    )

    logger.info(f"Q&A dataset generated and saved to: {output_file}")

    # Print summary
    print(f"\n{'='*60}")
    print("Q&A GENERATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total Questions: {len(qa_dataset)}")
    print(f"\nDistribution:")
    for qtype, count in distribution.items():
        actual = len([q for q in qa_dataset if q['question_type'] == qtype])
        print(f"  {qtype:15s}: {actual}/{count}")
    print(f"\nOutput File: {output_file}")
    print(f"{'='*60}\n")

    return output_file


def evaluate_rag_system(args):
    """Evaluate RAG system using Q&A dataset."""
    logger = get_logger(__name__)
    logger.info("Starting RAG system evaluation...")

    # Load Q&A dataset
    storage = QAStorage()
    qa_dataset = storage.load_dataset(args.dataset_file)

    logger.info(f"Loaded {len(qa_dataset)} Q&A pairs for evaluation")
    
    # Show original distribution
    original_distribution = storage.get_type_distribution(qa_dataset)
    logger.info(f"Original dataset distribution: {original_distribution}")

    # Apply stratified sampling when limiting questions
    if args.max_questions and args.max_questions < len(qa_dataset):
        num_types = len(original_distribution)
        samples_per_type = args.max_questions // num_types
        
        if samples_per_type > 0:
            logger.info(f"Using stratified sampling: {samples_per_type} questions per type (total: {samples_per_type * num_types})")
            qa_dataset = storage.sample_stratified_by_type(
                qa_dataset, 
                samples_per_type=samples_per_type,
                seed=42  # Fixed seed for reproducibility
            )
            sampled_distribution = storage.get_type_distribution(qa_dataset)
            logger.info(f"Sampled dataset distribution: {sampled_distribution}")
        else:
            logger.info(f"Limiting evaluation to {args.max_questions} questions (too small for stratified sampling)")
            qa_dataset = qa_dataset[:args.max_questions]

    # Determine evaluation mode
    answer_only_mode = getattr(args, 'answer_only', False)
    skip_answer = getattr(args, 'skip_answer_generation', False)
    include_answer_eval = not skip_answer

    logger.info(f"Evaluation mode - Answer only: {answer_only_mode}, Include answer eval: {include_answer_eval}, Skip answer: {skip_answer}")

    # Initialize evaluator with answer evaluation enabled by default
    evaluator = RAGEvaluator(enable_answer_evaluation=include_answer_eval)

    # Run evaluation
    results = evaluator.evaluate_dataset(
        qa_dataset,
        include_answer_generation=include_answer_eval,
        answer_only=answer_only_mode,
        save_individual_results=args.save_individual_results,
        max_answer_evaluations=getattr(args, 'max_answer_evaluations', None),
        max_retrieval_evaluations=getattr(args, 'max_retrieval_evaluations', None)
    )

    # Print summary
    print_evaluation_summary(results)

    # Save results
    output_file = storage.save_evaluation_results(
        results,
        filename=args.output_file
    )

    logger.info(f"Evaluation results saved to: {output_file}")

    # Generate HTML report
    report_path = os.path.join(os.path.dirname(output_file), 'evaluation_report.html')
    generate_html_report(results, args.dataset_file, report_path)
    logger.info(f"HTML report generated: {report_path}")

    print(f"\nReport saved to: {report_path}")

    return output_file


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="RAG System Evaluation - Generate Q&A pairs and evaluate retrieval performance"
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Generate Q&A dataset command
    gen_parser = subparsers.add_parser('generate', help='Generate Q&A dataset')
    gen_parser.add_argument(
        '--corpus-path',
        type=str,
        default=DEFAULT_CORPUS_PATH,
        help='Path to Wikipedia corpus JSON file'
    )
    gen_parser.add_argument(
        '--total-questions',
        type=int,
        default=DEFAULT_TOTAL_QUESTIONS,
        help='Total number of Q&A pairs to generate'
    )
    gen_parser.add_argument(
        '--equal-distribution',
        action='store_true',
        help='Distribute questions equally across all types'
    )
    gen_parser.add_argument(
        '--model-name',
        type=str,
        default=LLM_MODEL,
        help='LLM model name for question generation'
    )
    gen_parser.add_argument(
        '--output-file',
        type=str,
        default=None,
        help='Output filename for Q&A dataset (auto-generated if not specified)'
    )

    # Evaluate RAG system command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate RAG system')
    eval_parser.add_argument(
        '--dataset-file',
        type=str,
        required=True,
        help='Q&A dataset file to use for evaluation'
    )
    eval_parser.add_argument(
        '--max-questions', '--limit',
        type=int,
        default=None,
        dest='max_questions',
        help='Maximum number of questions to evaluate. Uses stratified sampling to '
             'ensure balanced representation across question types. '
             'Example: --max-questions 20 will select 5 from each of 4 question types.'
    )
    eval_parser.add_argument(
        '--skip-answer-generation',
        action='store_true',
        help='Skip answer generation and quality evaluation (faster, retrieval metrics only)'
    )
    eval_parser.add_argument(
        '--answer-only',
        action='store_true',
        help='Run only answer quality evaluation (skip retrieval metrics calculation)'
    )
    eval_parser.add_argument(
        '--max-answer-evaluations',
        type=int,
        default=None,
        help='Maximum number of questions to evaluate for answer quality (None = all)'
    )
    eval_parser.add_argument(
        '--max-retrieval-evaluations',
        type=int,
        default=None,
        help='Maximum number of questions to evaluate for retrieval quality (None = all)'
    )
    eval_parser.add_argument(
        '--save-individual-results',
        action='store_true',
        default=True,
        help='Save individual question results'
    )
    eval_parser.add_argument(
        '--output-file',
        type=str,
        default=None,
        help='Output filename for evaluation results'
    )

    # Full pipeline command
    full_parser = subparsers.add_parser('full', help='Generate Q&A and evaluate (full pipeline)')
    full_parser.add_argument(
        '--corpus-path',
        type=str,
        default=DEFAULT_CORPUS_PATH,
        help='Path to Wikipedia corpus JSON file'
    )
    full_parser.add_argument(
        '--total-questions',
        type=int,
        default=DEFAULT_TOTAL_QUESTIONS,
        help='Total number of Q&A pairs to generate'
    )
    full_parser.add_argument(
        '--equal-distribution',
        action='store_true',
        help='Distribute questions equally across all types'
    )
    full_parser.add_argument(
        '--max-eval-questions',
        type=int,
        default=None,
        help='Maximum number of questions to evaluate (for testing)'
    )
    full_parser.add_argument(
        '--skip-answer-generation',
        action='store_true',
        help='Skip answer generation and quality evaluation (faster, retrieval metrics only)'
    )
    full_parser.add_argument(
        '--answer-only',
        action='store_true',
        help='Run only answer quality evaluation (skip retrieval metrics calculation)'
    )

    args = parser.parse_args()

    if args.command == 'generate':
        generate_qa_dataset(args)

    elif args.command == 'evaluate':
        evaluate_rag_system(args)

    elif args.command == 'full':
        # Generate Q&A dataset
        print("\n" + "="*80)
        print("STEP 1: Generating Q&A Dataset")
        print("="*80 + "\n")

        gen_args = argparse.Namespace(
            corpus_path=args.corpus_path,
            total_questions=args.total_questions,
            equal_distribution=args.equal_distribution,
            model_name=LLM_MODEL,
            output_file=None
        )
        dataset_file = generate_qa_dataset(gen_args)

        # Evaluate RAG system
        print("\n" + "="*80)
        print("STEP 2: Evaluating RAG System")
        print("="*80 + "\n")

        eval_args = argparse.Namespace(
            dataset_file=dataset_file,
            max_questions=args.max_eval_questions,
            skip_answer_generation=args.skip_answer_generation,
            answer_only=getattr(args, 'answer_only', False),
            save_individual_results=True,
            output_file=None
        )
        evaluate_rag_system(eval_args)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
