"""
Q&A Dataset Storage and Validation
Handles saving, loading, and validating Q&A datasets
"""
import json
import os
import random
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import defaultdict
from src.util.logging_util import get_logger
from src.config.evaluation_config import (
    EVALUATION_STORAGE_DIR,
    QA_REQUIRED_FIELDS,
    QA_OPTIONAL_FIELDS,
    VALID_QUESTION_TYPES,
    VALID_DIFFICULTIES,
    DEFAULT_QA_DATASET_FILENAME
)


class QAStorage:
    """Manages Q&A dataset storage and validation."""

    REQUIRED_FIELDS = QA_REQUIRED_FIELDS
    OPTIONAL_FIELDS = QA_OPTIONAL_FIELDS
    VALID_QUESTION_TYPES = VALID_QUESTION_TYPES
    VALID_DIFFICULTIES = VALID_DIFFICULTIES

    def __init__(self, storage_dir: str = None):
        """
        Initialize QA storage.

        Args:
            storage_dir: Directory to store Q&A datasets (defaults to config)
        """
        self.logger = get_logger(__name__)
        self.storage_dir = storage_dir or EVALUATION_STORAGE_DIR
        os.makedirs(self.storage_dir, exist_ok=True)
        self.logger.info(f"QAStorage initialized with directory: {self.storage_dir}")

    def validate_qa_pair(self, qa_pair: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Validate a single Q&A pair.

        Args:
            qa_pair: Q&A pair dictionary

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check required fields
        for field in self.REQUIRED_FIELDS:
            if field not in qa_pair:
                return False, f"Missing required field: {field}"

            # Check non-empty
            value = qa_pair[field]
            if value is None or (isinstance(value, (str, list)) and len(value) == 0):
                return False, f"Required field '{field}' is empty"

        # Validate question_type
        if qa_pair['question_type'] not in self.VALID_QUESTION_TYPES:
            return False, f"Invalid question_type: {qa_pair['question_type']}. Must be one of {self.VALID_QUESTION_TYPES}"

        # Validate difficulty if present
        if 'difficulty' in qa_pair and qa_pair['difficulty'] not in self.VALID_DIFFICULTIES:
            return False, f"Invalid difficulty: {qa_pair['difficulty']}. Must be one of {self.VALID_DIFFICULTIES}"

        # Validate source_ids and source_urls are lists
        if not isinstance(qa_pair['source_ids'], list):
            return False, "source_ids must be a list"
        if not isinstance(qa_pair['source_urls'], list):
            return False, "source_urls must be a list"

        # Validate that lists have same length
        if len(qa_pair['source_ids']) != len(qa_pair['source_urls']):
            return False, "source_ids and source_urls must have the same length"

        return True, None

    def validate_dataset(self, qa_dataset: List[Dict[str, Any]]) -> tuple[bool, List[str]]:
        """
        Validate entire Q&A dataset.

        Args:
            qa_dataset: List of Q&A pairs

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        if not qa_dataset:
            return False, ["Dataset is empty"]

        errors = []
        question_ids = set()

        for i, qa_pair in enumerate(qa_dataset):
            # Validate individual pair
            is_valid, error = self.validate_qa_pair(qa_pair)
            if not is_valid:
                errors.append(f"Q&A pair {i+1}: {error}")

            # Check for duplicate question IDs
            qid = qa_pair.get('question_id')
            if qid in question_ids:
                errors.append(f"Duplicate question_id: {qid}")
            question_ids.add(qid)

        is_valid = len(errors) == 0
        return is_valid, errors

    def save_dataset(
        self,
        qa_dataset: List[Dict[str, Any]],
        filename: str = None,
        validate: bool = True,
        include_metadata: bool = True
    ) -> str:
        """
        Save Q&A dataset to JSON file.

        Args:
            qa_dataset: List of Q&A pairs
            filename: Output filename (auto-generated if None)
            validate: Whether to validate before saving
            include_metadata: Whether to include dataset metadata

        Returns:
            Path to saved file
        """
        # Validate if requested
        if validate:
            is_valid, errors = self.validate_dataset(qa_dataset)
            if not is_valid:
                error_msg = "\n".join(errors[:5])  # Show first 5 errors
                raise ValueError(f"Dataset validation failed:\n{error_msg}")

        # Generate filename if not provided
        if filename is None:
            filename = DEFAULT_QA_DATASET_FILENAME

        filepath = os.path.join(self.storage_dir, filename)

        # Prepare data structure
        data = {
            'total_questions': len(qa_dataset),
            'questions': qa_dataset
        }

        if include_metadata:
            # Add metadata
            question_types_count = {}
            difficulties_count = {}
            categories = set()

            for qa in qa_dataset:
                qtype = qa.get('question_type')
                question_types_count[qtype] = question_types_count.get(qtype, 0) + 1

                difficulty = qa.get('difficulty', 'unknown')
                difficulties_count[difficulty] = difficulties_count.get(difficulty, 0) + 1

                category = qa.get('category')
                if category:
                    categories.add(category)

            data['metadata'] = {
                'created_at': datetime.now().isoformat(),
                'question_types_distribution': question_types_count,
                'difficulty_distribution': difficulties_count,
                'categories': list(categories),
                'total_source_documents': len(set([
                    url for qa in qa_dataset for url in qa.get('source_urls', [])
                ]))
            }

        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Saved Q&A dataset with {len(qa_dataset)} questions to: {filepath}")
        return filepath

    def load_dataset(self, filename: str) -> List[Dict[str, Any]]:
        """
        Load Q&A dataset from JSON file.

        Args:
            filename: Input filename or full path

        Returns:
            List of Q&A pairs
        """
        # Check if it's a full path or just filename
        if os.path.exists(filename):
            filepath = filename
        else:
            filepath = os.path.join(self.storage_dir, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        qa_dataset = data.get('questions', [])
        self.logger.info(f"Loaded Q&A dataset with {len(qa_dataset)} questions from: {filepath}")

        return qa_dataset

    def save_evaluation_results(
        self,
        results: Dict[str, Any],
        filename: str = None
    ) -> str:
        """
        Save evaluation results to JSON file.

        Args:
            results: Evaluation results dictionary
            filename: Output filename (auto-generated if None)

        Returns:
            Path to saved file
        """
        if filename is None:
            filename = "evaluation_results.json"

        filepath = os.path.join(self.storage_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Saved evaluation results to: {filepath}")
        return filepath

    def list_datasets(self) -> List[str]:
        """
        List all available Q&A datasets.

        Returns:
            List of dataset filenames
        """
        if not os.path.exists(self.storage_dir):
            return []

        datasets = [
            f for f in os.listdir(self.storage_dir)
            if f.startswith('qa_dataset_') and f.endswith('.json')
        ]
        return sorted(datasets)

    def list_evaluation_results(self) -> List[str]:
        """
        List all available evaluation result files.

        Returns:
            List of evaluation result filenames
        """
        if not os.path.exists(self.storage_dir):
            return []

        results = [
            f for f in os.listdir(self.storage_dir)
            if f.startswith('evaluation_results_') and f.endswith('.json')
        ]
        return sorted(results)

    def get_dataset_info(self, filename: str) -> Dict[str, Any]:
        """
        Get metadata information about a dataset without loading all questions.

        Args:
            filename: Dataset filename

        Returns:
            Dataset metadata
        """
        filepath = os.path.join(self.storage_dir, filename)

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return {
            'filename': filename,
            'total_questions': data.get('total_questions', 0),
            'metadata': data.get('metadata', {})
        }

    def merge_datasets(
        self,
        dataset_files: List[str],
        output_filename: str = None
    ) -> str:
        """
        Merge multiple Q&A datasets into one.

        Args:
            dataset_files: List of dataset filenames to merge
            output_filename: Output filename for merged dataset

        Returns:
            Path to merged dataset file
        """
        self.logger.info(f"Merging {len(dataset_files)} datasets...")

        all_questions = []
        question_id_counter = 1

        for dataset_file in dataset_files:
            questions = self.load_dataset(dataset_file)

            # Renumber question IDs to avoid conflicts
            for qa in questions:
                qa['question_id'] = f"Q{question_id_counter:03d}"
                question_id_counter += 1

            all_questions.extend(questions)

        # Save merged dataset
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"qa_dataset_merged_{timestamp}.json"

        filepath = self.save_dataset(all_questions, output_filename)
        self.logger.info(f"Merged dataset saved with {len(all_questions)} questions")

        return filepath

    def export_to_csv(self, dataset_file: str, output_csv: str = None) -> str:
        """
        Export Q&A dataset to CSV format.

        Args:
            dataset_file: Input dataset filename
            output_csv: Output CSV filename

        Returns:
            Path to CSV file
        """
        import csv

        qa_dataset = self.load_dataset(dataset_file)

        if output_csv is None:
            output_csv = dataset_file.replace('.json', '.csv')

        csv_path = os.path.join(self.storage_dir, output_csv)

        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['question_id', 'question', 'answer', 'question_type',
                         'difficulty', 'source_urls', 'category']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for qa in qa_dataset:
                row = {
                    'question_id': qa['question_id'],
                    'question': qa['question'],
                    'answer': qa['answer'],
                    'question_type': qa['question_type'],
                    'difficulty': qa.get('difficulty', 'medium'),
                    'source_urls': '|'.join(qa.get('source_urls', [])),
                    'category': qa.get('category', '')
                }
                writer.writerow(row)

        self.logger.info(f"Exported dataset to CSV: {csv_path}")
        return csv_path

    def sample_stratified_by_type(
        self,
        qa_dataset: List[Dict[str, Any]],
        samples_per_type: int = 5,
        seed: int = None
    ) -> List[Dict[str, Any]]:
        """
        Sample questions with stratified sampling by question type.

        This ensures equal representation of all question types in the sample.
        For example, with 4 question types and samples_per_type=5, you get
        exactly 5 questions from each type for a total of 20 questions.

        Args:
            qa_dataset: List of Q&A pairs
            samples_per_type: Number of samples to take from each question type
            seed: Random seed for reproducibility (optional)

        Returns:
            Stratified sample of Q&A pairs

        Raises:
            ValueError: If there aren't enough questions of a particular type
        """
        if seed is not None:
            random.seed(seed)

        # Group questions by type
        questions_by_type = defaultdict(list)
        for qa in qa_dataset:
            qtype = qa.get('question_type')
            if qtype:
                questions_by_type[qtype].append(qa)

        # Check if we have enough questions of each type
        available_types = list(questions_by_type.keys())
        for qtype in available_types:
            count = len(questions_by_type[qtype])
            if count < samples_per_type:
                self.logger.warning(
                    f"Question type '{qtype}' has only {count} questions, "
                    f"but {samples_per_type} were requested"
                )

        # Sample from each type
        stratified_sample = []
        for qtype in sorted(available_types):  # Sort for deterministic order
            available = questions_by_type[qtype]
            sample_size = min(samples_per_type, len(available))
            sampled = random.sample(available, sample_size)
            stratified_sample.extend(sampled)
            self.logger.info(f"Sampled {sample_size} questions of type '{qtype}'")

        # Shuffle the final sample to mix question types
        random.shuffle(stratified_sample)

        self.logger.info(
            f"Stratified sampling complete: {len(stratified_sample)} total questions "
            f"({samples_per_type} per type × {len(available_types)} types)"
        )

        return stratified_sample

    def get_type_distribution(self, qa_dataset: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Get the distribution of question types in the dataset.

        Args:
            qa_dataset: List of Q&A pairs

        Returns:
            Dictionary mapping question types to their counts
        """
        distribution = defaultdict(int)
        for qa in qa_dataset:
            qtype = qa.get('question_type', 'unknown')
            distribution[qtype] += 1

        return dict(distribution)
