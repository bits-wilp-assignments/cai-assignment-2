"""
RAG System Evaluator
Implements evaluation metrics for RAG system including MRR, retrieval accuracy, and answer quality
"""
import numpy as np
import re
from typing import List, Dict, Any, Tuple
from collections import defaultdict, Counter
from src.util.logging_util import get_logger
from src.service.retrieval import hybrid_retriever
from src.service.inference import rag_inference
from src.core.embedding import EmbeddingFactory
from src.config.app_config import EMBEDDING_MODEL, IS_NORM_EMBEDDINGS_ENABLED, MODEL_LOCAL_FILES_ONLY
from src.config.evaluation_config import (
    EVALUATION_RETRIEVAL_TOP_K,
    EVALUATION_K_VALUES,
    ENABLE_ANSWER_EVALUATION,
    DEFAULT_MAX_RETRIEVAL_EVALUATIONS
)


class RAGEvaluator:
    """Evaluates RAG system performance using various metrics."""

    def __init__(self, enable_answer_evaluation: bool = None):
        """
        Initialize evaluator.

        Args:
            enable_answer_evaluation: Whether to evaluate answer quality (defaults to config)
        """
        if enable_answer_evaluation is None:
            enable_answer_evaluation = ENABLE_ANSWER_EVALUATION
        self.logger = get_logger(__name__)
        self.enable_answer_evaluation = enable_answer_evaluation

        # Initialize embeddings for answer similarity if enabled
        if enable_answer_evaluation:
            try:
                embedding_factory = EmbeddingFactory()
                self.embedding_model = embedding_factory.get_instance(
                    model_name=EMBEDDING_MODEL,
                    norm_embeddings=IS_NORM_EMBEDDINGS_ENABLED,
                    local_files_only=MODEL_LOCAL_FILES_ONLY
                )
                self.logger.info("Answer evaluation enabled with embeddings")
            except Exception as e:
                self.logger.warning(f"Could not load embeddings for answer evaluation: {e}")
                self.enable_answer_evaluation = False

        self.logger.info("RAGEvaluator initialized")

    def _extract_url_from_doc(self, doc) -> str:
        """
        Extract Wikipedia URL from document metadata.

        Args:
            doc: Retrieved document

        Returns:
            URL string or empty string if not found
        """
        if hasattr(doc, 'metadata'):
            metadata = doc.metadata
            # Check for URL in different possible fields
            return metadata.get('url', metadata.get('source', metadata.get('page_url', '')))
        return ''

    def calculate_mrr_at_url_level(
        self,
        question: str,
        ground_truth_urls: List[str],
        top_k: int = None
    ) -> Tuple[float, int, List[str]]:
        """
        Calculate Mean Reciprocal Rank at URL level.

        For each question, find the rank position of the first correct Wikipedia URL
        in the retrieved results. MRR = 1/rank where rank is the position of the
        first correct URL.

        Args:
            question: Query question
            ground_truth_urls: List of correct Wikipedia URLs for this question
            top_k: Number of results to retrieve

        Returns:
            Tuple of (reciprocal_rank, rank_of_first_correct, retrieved_urls)
        """
        if top_k is None:
            top_k = EVALUATION_RETRIEVAL_TOP_K

        self.logger.info(f"Calculating MRR for question: {question[:50]}...")

        # Retrieve documents
        try:
            results = hybrid_retriever.retrieve(
                query=question,
                dense_top_k=top_k,
                sparse_top_k=top_k,
                rrf_top_k=top_k,
                reranker_top_k=top_k
            )

            # Extract URLs from retrieved documents
            retrieved_urls = []
            for doc_map in results:
                doc = doc_map.get('doc')
                url = self._extract_url_from_doc(doc)
                if url:
                    retrieved_urls.append(url)

            self.logger.debug(f"Retrieved URLs: {retrieved_urls}")
            self.logger.debug(f"Ground truth URLs: {ground_truth_urls}")

            # Find rank of first correct URL
            rank_of_first_correct = -1
            for rank, url in enumerate(retrieved_urls, start=1):
                # Check if this URL matches any ground truth URL
                for gt_url in ground_truth_urls:
                    if gt_url in url or url in gt_url:  # Flexible matching
                        rank_of_first_correct = rank
                        break
                if rank_of_first_correct > 0:
                    break

            # Calculate reciprocal rank
            if rank_of_first_correct > 0:
                reciprocal_rank = 1.0 / rank_of_first_correct
                self.logger.info(f"First correct URL found at rank {rank_of_first_correct}, RR={reciprocal_rank:.4f}")
            else:
                reciprocal_rank = 0.0
                self.logger.warning(f"No correct URL found in top {top_k} results")

            return reciprocal_rank, rank_of_first_correct, retrieved_urls

        except Exception as e:
            self.logger.error(f"Error calculating MRR: {e}")
            return 0.0, -1, []

    def calculate_answer_similarity(self, generated_answer: str, ground_truth_answer: str) -> float:
        """
        Calculate semantic similarity between generated and ground truth answers using embeddings.

        Args:
            generated_answer: Answer generated by RAG system
            ground_truth_answer: Ground truth answer

        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        if not self.enable_answer_evaluation or not generated_answer or not ground_truth_answer:
            return 0.0

        try:
            # Generate embeddings
            gen_embedding = self.embedding_model.embed_query(generated_answer)
            gt_embedding = self.embedding_model.embed_query(ground_truth_answer)

            # Calculate cosine similarity
            similarity = np.dot(gen_embedding, gt_embedding) / (
                np.linalg.norm(gen_embedding) * np.linalg.norm(gt_embedding)
            )

            return float(similarity)
        except Exception as e:
            self.logger.error(f"Error calculating answer similarity: {e}")
            return 0.0

    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = ' '.join(text.split())
        return text

    def calculate_f1_score(self, generated_answer: str, ground_truth_answer: str) -> float:
        """
        Calculate token-level F1 score.

        Args:
            generated_answer: Answer generated by RAG system
            ground_truth_answer: Ground truth answer

        Returns:
            F1 score (0.0 to 1.0)
        """
        if not generated_answer or not ground_truth_answer:
            return 0.0

        gen_tokens = self.normalize_text(generated_answer).split()
        gt_tokens = self.normalize_text(ground_truth_answer).split()

        if not gen_tokens or not gt_tokens:
            return 0.0

        common = Counter(gen_tokens) & Counter(gt_tokens)
        num_common = sum(common.values())

        if num_common == 0:
            return 0.0

        precision = num_common / len(gen_tokens)
        recall = num_common / len(gt_tokens)
        f1 = 2 * (precision * recall) / (precision + recall)

        return f1

    def calculate_bleu_score(self, generated_answer: str, ground_truth_answer: str) -> float:
        """
        Calculate BLEU score (simple unigram version).

        Args:
            generated_answer: Answer generated by RAG system
            ground_truth_answer: Ground truth answer

        Returns:
            BLEU score (0.0 to 1.0)
        """
        if not generated_answer or not ground_truth_answer:
            return 0.0

        try:
            gen_tokens = self.normalize_text(generated_answer).split()
            gt_tokens = self.normalize_text(ground_truth_answer).split()

            if not gen_tokens or not gt_tokens:
                return 0.0

            # Simple unigram precision
            common = Counter(gen_tokens) & Counter(gt_tokens)
            num_common = sum(common.values())
            precision = num_common / len(gen_tokens) if gen_tokens else 0.0

            # Brevity penalty
            bp = 1.0 if len(gen_tokens) > len(gt_tokens) else np.exp(1 - len(gt_tokens) / len(gen_tokens))

            return bp * precision
        except Exception as e:
            self.logger.error(f"Error calculating BLEU: {e}")
            return 0.0

    def calculate_rouge_l(self, generated_answer: str, ground_truth_answer: str) -> float:
        """
        Calculate ROUGE-L score (Longest Common Subsequence).

        Args:
            generated_answer: Answer generated by RAG system
            ground_truth_answer: Ground truth answer

        Returns:
            ROUGE-L F1 score (0.0 to 1.0)
        """
        if not generated_answer or not ground_truth_answer:
            return 0.0

        try:
            gen_tokens = self.normalize_text(generated_answer).split()
            gt_tokens = self.normalize_text(ground_truth_answer).split()

            if not gen_tokens or not gt_tokens:
                return 0.0

            # Calculate LCS length
            m, n = len(gen_tokens), len(gt_tokens)
            dp = [[0] * (n + 1) for _ in range(m + 1)]

            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if gen_tokens[i-1] == gt_tokens[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])

            lcs_length = dp[m][n]

            if lcs_length == 0:
                return 0.0

            # Calculate precision and recall
            precision = lcs_length / len(gen_tokens)
            recall = lcs_length / len(gt_tokens)

            # Calculate F1
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            return f1
        except Exception as e:
            self.logger.error(f"Error calculating ROUGE-L: {e}")
            return 0.0

    def calculate_answer_quality_metrics(self, generated_answer: str, ground_truth_answer: str) -> Dict[str, float]:
        """
        Calculate all answer quality metrics.

        Args:
            generated_answer: Answer generated by RAG system
            ground_truth_answer: Ground truth answer

        Returns:
            Dictionary with all answer quality metrics
        """
        metrics = {
            'f1_score': self.calculate_f1_score(generated_answer, ground_truth_answer),
            'bleu_score': self.calculate_bleu_score(generated_answer, ground_truth_answer),
            'rouge_l': self.calculate_rouge_l(generated_answer, ground_truth_answer),
            'semantic_similarity': self.calculate_answer_similarity(generated_answer, ground_truth_answer)
        }

        return metrics

    def calculate_retrieval_precision_at_k(
        self,
        question: str,
        ground_truth_urls: List[str],
        k: int = 5
    ) -> float:
        """
        Calculate Precision@K for retrieval.

        Precision@K = (Number of relevant documents in top K) / K

        Args:
            question: Query question
            ground_truth_urls: List of correct Wikipedia URLs
            k: Number of top results to consider

        Returns:
            Precision@K score
        """
        try:
            results = hybrid_retriever.retrieve(
                query=question,
                dense_top_k=k,
                sparse_top_k=k,
                rrf_top_k=k,
                reranker_top_k=k
            )

            retrieved_urls = [self._extract_url_from_doc(doc_map.get('doc')) for doc_map in results]

            # Count relevant documents in top K
            relevant_count = 0
            for url in retrieved_urls[:k]:
                for gt_url in ground_truth_urls:
                    if gt_url in url or url in gt_url:
                        relevant_count += 1
                        break

            precision = relevant_count / k
            self.logger.debug(f"Precision@{k}: {precision:.4f} ({relevant_count}/{k})")
            return precision

        except Exception as e:
            self.logger.error(f"Error calculating Precision@K: {e}")
            return 0.0

    def calculate_retrieval_recall_at_k(
        self,
        question: str,
        ground_truth_urls: List[str],
        k: int = 10
    ) -> float:
        """
        Calculate Recall@K for retrieval.

        Recall@K = (Number of relevant documents retrieved in top K) / (Total relevant documents)

        Args:
            question: Query question
            ground_truth_urls: List of correct Wikipedia URLs
            k: Number of top results to consider

        Returns:
            Recall@K score
        """
        if not ground_truth_urls:
            return 0.0

        try:
            results = hybrid_retriever.retrieve(
                query=question,
                dense_top_k=k,
                sparse_top_k=k,
                rrf_top_k=k,
                reranker_top_k=k
            )

            retrieved_urls = [self._extract_url_from_doc(doc_map.get('doc')) for doc_map in results]

            # Count how many ground truth URLs are in retrieved results
            found_count = 0
            for gt_url in ground_truth_urls:
                for url in retrieved_urls[:k]:
                    if gt_url in url or url in gt_url:
                        found_count += 1
                        break

            recall = found_count / len(ground_truth_urls)
            self.logger.debug(f"Recall@{k}: {recall:.4f} ({found_count}/{len(ground_truth_urls)})")
            return recall

        except Exception as e:
            self.logger.error(f"Error calculating Recall@K: {e}")
            return 0.0

    def calculate_hit_rate_at_k(
        self,
        question: str,
        ground_truth_urls: List[str],
        k: int = 10
    ) -> float:
        """
        Calculate Hit Rate@K.

        Hit Rate@K = 1 if at least one relevant document is in top K, else 0

        Args:
            question: Query question
            ground_truth_urls: List of correct Wikipedia URLs
            k: Number of top results to consider

        Returns:
            Hit rate (0 or 1)
        """
        try:
            results = hybrid_retriever.retrieve(
                query=question,
                dense_top_k=k,
                sparse_top_k=k,
                rrf_top_k=k,
                reranker_top_k=k
            )

            retrieved_urls = [self._extract_url_from_doc(doc_map.get('doc')) for doc_map in results]

            # Check if any ground truth URL is in retrieved results
            for gt_url in ground_truth_urls:
                for url in retrieved_urls[:k]:
                    if gt_url in url or url in gt_url:
                        return 1.0

            return 0.0

        except Exception as e:
            self.logger.error(f"Error calculating Hit Rate@K: {e}")
            return 0.0

    def evaluate_single_qa(
        self,
        qa_pair: Dict[str, Any],
        include_answer_generation: bool = False,
        answer_only: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate a single Q&A pair.

        Args:
            qa_pair: Q&A pair dictionary
            include_answer_generation: Whether to generate and evaluate answer
            answer_only: If True, skip retrieval metrics calculation

        Returns:
            Evaluation results dictionary
        """
        question = qa_pair['question']
        ground_truth_urls = qa_pair.get('source_urls', [])
        question_id = qa_pair.get('question_id', 'unknown')

        self.logger.info(f"Evaluating {question_id}: {question[:60]}...")

        results = {
            'question_id': question_id,
            'question': question,
            'question_type': qa_pair.get('question_type'),
            'ground_truth_urls': ground_truth_urls
        }

        # Retrieval metrics (skip if answer_only mode)
        if not answer_only:
            try:
                # MRR at URL level
                rr, rank, retrieved_urls = self.calculate_mrr_at_url_level(question, ground_truth_urls, top_k=10)
                results['reciprocal_rank'] = rr
                results['rank_of_first_correct'] = rank
                results['retrieved_urls'] = retrieved_urls

                # Precision, Recall, Hit Rate at different K values
                for k in [3, 5, 10]:
                    results[f'precision_at_{k}'] = self.calculate_retrieval_precision_at_k(question, ground_truth_urls, k)
                    results[f'recall_at_{k}'] = self.calculate_retrieval_recall_at_k(question, ground_truth_urls, k)
                    results[f'hit_rate_at_{k}'] = self.calculate_hit_rate_at_k(question, ground_truth_urls, k)

            except Exception as e:
                self.logger.error(f"Error calculating retrieval metrics: {e}")

        # Answer generation and evaluation
        if include_answer_generation:
            try:
                generated_answer = rag_inference(question)

                # Convert generator to string if necessary
                if hasattr(generated_answer, '__iter__') and not isinstance(generated_answer, str):
                    generated_answer = ''.join(str(chunk) for chunk in generated_answer)

                ground_truth_answer = qa_pair.get('answer')
                results['generated_answer'] = generated_answer
                results['ground_truth_answer'] = ground_truth_answer

                # Calculate all answer quality metrics if enabled
                if self.enable_answer_evaluation and generated_answer and ground_truth_answer:
                    quality_metrics = self.calculate_answer_quality_metrics(generated_answer, ground_truth_answer)

                    # Add all metrics to results
                    results['f1_score'] = quality_metrics['f1_score']
                    results['bleu_score'] = quality_metrics['bleu_score']
                    results['rouge_l'] = quality_metrics['rouge_l']
                    results['semantic_similarity'] = quality_metrics['semantic_similarity']

                    self.logger.info(f"Answer quality - F1: {quality_metrics['f1_score']:.4f}, "
                                   f"Semantic: {quality_metrics['semantic_similarity']:.4f}")
            except Exception as e:
                self.logger.error(f"Error generating/evaluating answer: {e}")
                results['generated_answer'] = None
                results['f1_score'] = 0.0
                results['bleu_score'] = 0.0
                results['rouge_l'] = 0.0
                results['semantic_similarity'] = 0.0

        return results

    def _apply_stratified_sampling(
        self,
        qa_dataset: List[Dict[str, Any]],
        max_answer_evaluations: int = None,
        max_retrieval_evaluations: int = None,
        include_answer_generation: bool = False,
        answer_only: bool = False
    ) -> tuple[List[Dict[str, Any]], Dict[str, Dict[str, int]]]:
        """
        Apply stratified sampling to ensure balanced evaluation across question types.

        Args:
            qa_dataset: List of Q&A pairs
            max_answer_evaluations: Maximum number for answer evaluation
            max_retrieval_evaluations: Maximum number for retrieval evaluation
            include_answer_generation: Whether answer evaluation is enabled
            answer_only: Whether in answer-only mode

        Returns:
            Tuple of (sampled questions to evaluate, evaluation plan by type)
        """
        # Group questions by type
        questions_by_type = defaultdict(list)
        for qa in qa_dataset:
            qtype = qa.get('question_type', 'unknown')
            questions_by_type[qtype].append(qa)

        num_types = len(questions_by_type)
        if num_types == 0:
            return qa_dataset, {}

        # Determine the maximum limit to use for sampling
        # Use the larger of the two limits to ensure we have enough questions for both evaluations
        limits = []
        if include_answer_generation and max_answer_evaluations is not None:
            limits.append(max_answer_evaluations)
        if not answer_only and max_retrieval_evaluations is not None:
            limits.append(max_retrieval_evaluations)

        active_limit = max(limits) if limits else None

        # If no limit, return original dataset
        if active_limit is None or active_limit >= len(qa_dataset):
            return qa_dataset, {}

        # Calculate samples per type (stratified)
        samples_per_type = max(1, active_limit // num_types)

        # Sample from each type
        sampled_questions = []
        eval_plan = {}

        for qtype, questions in sorted(questions_by_type.items()):
            # Take up to samples_per_type from this type
            sample_size = min(samples_per_type, len(questions))
            sampled = questions[:sample_size]
            sampled_questions.extend(sampled)
            eval_plan[qtype] = {
                'total_available': len(questions),
                'to_evaluate': sample_size
            }

        self.logger.info(f"Stratified sampling applied: {len(sampled_questions)} questions selected")
        for qtype, plan in eval_plan.items():
            self.logger.info(f"  {qtype}: {plan['to_evaluate']}/{plan['total_available']}")

        return sampled_questions, eval_plan

    def evaluate_dataset(
        self,
        qa_dataset: List[Dict[str, Any]],
        include_answer_generation: bool = False,
        answer_only: bool = False,
        save_individual_results: bool = True,
        max_answer_evaluations: int = None,
        max_retrieval_evaluations: int = None
    ) -> Dict[str, Any]:
        """
        Evaluate entire Q&A dataset with stratified sampling across question types.

        Args:
            qa_dataset: List of Q&A pairs
            include_answer_generation: Whether to generate and evaluate answers
            answer_only: If True, skip retrieval metrics (only evaluate answer quality)
            save_individual_results: Whether to include individual results in output
            max_answer_evaluations: Maximum number of questions to evaluate for answer quality (None = all)
            max_retrieval_evaluations: Maximum number of questions to evaluate for retrieval quality (None = all)

        Returns:
            Comprehensive evaluation results
        """
        self.logger.info(f"Starting dataset evaluation: {len(qa_dataset)} questions")
        if answer_only:
            self.logger.info("Answer-only mode: Skipping retrieval metrics calculation")
        if max_answer_evaluations is not None and include_answer_generation:
            self.logger.info(f"Answer quality evaluation limited to {max_answer_evaluations} questions")
        if max_retrieval_evaluations is not None and not answer_only:
            self.logger.info(f"Retrieval quality evaluation limited to {max_retrieval_evaluations} questions")

        # Apply stratified sampling if limits are specified
        questions_to_evaluate, eval_plan = self._apply_stratified_sampling(
            qa_dataset,
            max_answer_evaluations,
            max_retrieval_evaluations,
            include_answer_generation,
            answer_only
        )

        individual_results = []

        # Aggregate metrics
        total_rr = 0.0
        total_exact_match = 0.0
        total_f1 = 0.0
        total_bleu = 0.0
        total_rouge = 0.0
        total_semantic_sim = 0.0
        answer_count = 0
        retrieval_count = 0

        # Track counts per question type for stratified limits
        answer_count_by_type = defaultdict(int)
        retrieval_count_by_type = defaultdict(int)

        metrics_by_type = defaultdict(lambda: {
            'count': 0,
            'total_rr': 0.0,
            'precision_at_3': [],
            'precision_at_5': [],
            'precision_at_10': [],
            'recall_at_3': [],
            'recall_at_5': [],
            'recall_at_10': [],
            'hit_rate_at_3': [],
            'hit_rate_at_5': [],
            'hit_rate_at_10': [],
            'f1_scores': [],
            'bleu_scores': [],
            'rouge_l_scores': [],
            'semantic_similarities': []
        })

        # Calculate per-type limits for stratified sampling
        questions_by_type = defaultdict(list)
        for qa in questions_to_evaluate:
            qtype = qa.get('question_type', 'unknown')
            questions_by_type[qtype].append(qa)

        num_types = len(questions_by_type)
        answer_limit_per_type = None
        retrieval_limit_per_type = None

        if max_answer_evaluations is not None and include_answer_generation and num_types > 0:
            answer_limit_per_type = max(1, max_answer_evaluations // num_types)
        if max_retrieval_evaluations is not None and not answer_only and num_types > 0:
            retrieval_limit_per_type = max(1, max_retrieval_evaluations // num_types)

        for i, qa_pair in enumerate(questions_to_evaluate, 1):
            question_type = qa_pair.get('question_type', 'unknown')

            # Apply per-evaluation-type and per-question-type limits (stratified)
            should_evaluate_retrieval = (not answer_only) and (
                retrieval_limit_per_type is None or
                retrieval_count_by_type[question_type] < retrieval_limit_per_type
            )
            should_evaluate_answer = include_answer_generation and (
                answer_limit_per_type is None or
                answer_count_by_type[question_type] < answer_limit_per_type
            )

            # Skip if both evaluations are disabled for this question
            if not should_evaluate_retrieval and not should_evaluate_answer:
                continue

            # Determine the actual answer_only mode for this question
            effective_answer_only = answer_only or not should_evaluate_retrieval

            result = self.evaluate_single_qa(qa_pair, should_evaluate_answer, effective_answer_only)
            individual_results.append(result)

            # Track that we've seen this question type
            if question_type not in metrics_by_type:
                metrics_by_type[question_type]  # Initialize with default values

            # Only aggregate retrieval metrics if not in answer_only mode
            if not effective_answer_only and 'reciprocal_rank' in result:
                rr = result.get('reciprocal_rank', 0.0)
                total_rr += rr
                metrics_by_type[question_type]['total_rr'] += rr
                retrieval_count += 1
                retrieval_count_by_type[question_type] += 1

                # Aggregate precision, recall, hit rate
                for k in [3, 5, 10]:
                    metrics_by_type[question_type][f'precision_at_{k}'].append(result.get(f'precision_at_{k}', 0.0))
                    metrics_by_type[question_type][f'recall_at_{k}'].append(result.get(f'recall_at_{k}', 0.0))
                    metrics_by_type[question_type][f'hit_rate_at_{k}'].append(result.get(f'hit_rate_at_{k}', 0.0))

            # Aggregate answer quality metrics if available
            if 'f1_score' in result:
                total_f1 += result.get('f1_score', 0.0)
                total_bleu += result.get('bleu_score', 0.0)
                total_rouge += result.get('rouge_l', 0.0)
                total_semantic_sim += result.get('semantic_similarity', 0.0)
                answer_count += 1
                answer_count_by_type[question_type] += 1

                metrics_by_type[question_type]['f1_scores'].append(result.get('f1_score', 0.0))
                metrics_by_type[question_type]['bleu_scores'].append(result.get('bleu_score', 0.0))
                metrics_by_type[question_type]['rouge_l_scores'].append(result.get('rouge_l', 0.0))
                metrics_by_type[question_type]['semantic_similarities'].append(result.get('semantic_similarity', 0.0))

            if i % 10 == 0:
                self.logger.info(f"Evaluated {i}/{len(questions_to_evaluate)} questions")

        # Calculate final metrics
        n = len(qa_dataset)
        n_evaluated = len(questions_to_evaluate)
        overall_mrr = total_rr / retrieval_count if retrieval_count > 0 and not answer_only else None

        # Calculate overall answer quality metrics
        if answer_count > 0:
            overall_f1 = total_f1 / answer_count
            overall_bleu = total_bleu / answer_count
            overall_rouge = total_rouge / answer_count
            overall_semantic_sim = total_semantic_sim / answer_count
        else:
            overall_f1 = None
            overall_bleu = None
            overall_rouge = None
            overall_semantic_sim = None

        summary = {
            'total_questions': n,
            'questions_evaluated': n_evaluated,
            'questions_evaluated_retrieval': retrieval_count,
            'questions_evaluated_answer': answer_count,
            'overall_mrr': overall_mrr,
            'overall_f1': overall_f1,
            'overall_bleu': overall_bleu,
            'overall_rouge_l': overall_rouge,
            'overall_semantic_similarity': overall_semantic_sim,
            'answer_evaluation_enabled': include_answer_generation and self.enable_answer_evaluation,
            'answer_only_mode': answer_only,
            'metrics_by_question_type': {}
        }

        # Calculate per-type metrics
        for qtype, metrics in metrics_by_type.items():
            # Count based on retrieval metrics (precision lists) or answer metrics (f1_scores)
            retrieval_count_for_type = len(metrics['precision_at_5']) if metrics['precision_at_5'] else 0
            answer_count_for_type = len(metrics['f1_scores']) if metrics['f1_scores'] else 0

            # Use whichever count is available (or max if both)
            count = max(retrieval_count_for_type, answer_count_for_type)

            if count > 0:
                type_summary = {'count': count}

                # Add retrieval metrics if not in answer_only mode and we have retrieval data
                if not answer_only and retrieval_count_for_type > 0:
                    type_summary['mrr'] = metrics['total_rr'] / retrieval_count_for_type
                    type_summary['precision_at_3'] = np.mean(metrics['precision_at_3'])
                    type_summary['precision_at_5'] = np.mean(metrics['precision_at_5'])
                    type_summary['precision_at_10'] = np.mean(metrics['precision_at_10'])
                    type_summary['recall_at_3'] = np.mean(metrics['recall_at_3'])
                    type_summary['recall_at_5'] = np.mean(metrics['recall_at_5'])
                    type_summary['recall_at_10'] = np.mean(metrics['recall_at_10'])
                    type_summary['hit_rate_at_3'] = np.mean(metrics['hit_rate_at_3'])
                    type_summary['hit_rate_at_5'] = np.mean(metrics['hit_rate_at_5'])
                    type_summary['hit_rate_at_10'] = np.mean(metrics['hit_rate_at_10'])

                # Add answer quality metrics if available
                if metrics['f1_scores']:
                    type_summary['f1_score'] = np.mean(metrics['f1_scores'])
                    type_summary['bleu_score'] = np.mean(metrics['bleu_scores'])
                    type_summary['rouge_l'] = np.mean(metrics['rouge_l_scores'])
                    type_summary['semantic_similarity'] = np.mean(metrics['semantic_similarities'])
                summary['metrics_by_question_type'][qtype] = type_summary

        # Calculate overall averages (only if not in answer_only mode)
        if not answer_only:
            all_precision_3 = [r.get('precision_at_3', 0.0) for r in individual_results]
            all_precision_5 = [r.get('precision_at_5', 0.0) for r in individual_results]
            all_precision_10 = [r.get('precision_at_10', 0.0) for r in individual_results]
            all_recall_3 = [r.get('recall_at_3', 0.0) for r in individual_results]
            all_recall_5 = [r.get('recall_at_5', 0.0) for r in individual_results]
            all_recall_10 = [r.get('recall_at_10', 0.0) for r in individual_results]
            all_hit_3 = [r.get('hit_rate_at_3', 0.0) for r in individual_results]
            all_hit_5 = [r.get('hit_rate_at_5', 0.0) for r in individual_results]
            all_hit_10 = [r.get('hit_rate_at_10', 0.0) for r in individual_results]

            summary['overall_metrics'] = {
                'precision_at_3': np.mean(all_precision_3),
                'precision_at_5': np.mean(all_precision_5),
                'precision_at_10': np.mean(all_precision_10),
                'recall_at_3': np.mean(all_recall_3),
                'recall_at_5': np.mean(all_recall_5),
                'recall_at_10': np.mean(all_recall_10),
                'hit_rate_at_3': np.mean(all_hit_3),
                'hit_rate_at_5': np.mean(all_hit_5),
                'hit_rate_at_10': np.mean(all_hit_10)
            }

        log_msg = f"Evaluation complete."
        if not answer_only and overall_mrr is not None:
            log_msg += f" Overall MRR: {overall_mrr:.4f}"
        if overall_semantic_sim is not None:
            log_msg += f" Overall Semantic Similarity: {overall_semantic_sim:.4f}"
            log_msg += f" | F1: {overall_f1:.4f} | BLEU: {overall_bleu:.4f} | ROUGE-L: {overall_rouge:.4f}"
        self.logger.info(log_msg)

        result = {'summary': summary}
        if save_individual_results:
            result['individual_results'] = individual_results

        return result
