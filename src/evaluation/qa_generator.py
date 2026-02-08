"""
Q&A Generator for RAG Evaluation
Generates diverse question-answer pairs from Wikipedia corpus using LLM
"""
import json
import random
from typing import List, Dict, Any
from src.core.llm import LLMFactory
from src.util.logging_util import get_logger
from src.config.app_config import LLM_MODEL


class QAGenerator:
    """Generates Q&A pairs from Wikipedia documents using LLM."""

    QUESTION_TYPES = {
        "factual": {
            "description": "Direct factual questions about specific information",
            "examples": [
                "What is X?",
                "When did Y occur?",
                "Who was Z?",
                "Where is A located?"
            ]
        },
        "comparative": {
            "description": "Questions comparing two or more concepts",
            "examples": [
                "What is the difference between X and Y?",
                "How does X compare to Y?",
                "What are the similarities between A and B?"
            ]
        },
        "inferential": {
            "description": "Questions requiring reasoning and inference",
            "examples": [
                "Why does X cause Y?",
                "How does A affect B?",
                "What are the implications of X?",
                "What can be inferred about Y from X?"
            ]
        },
        "multi-hop": {
            "description": "Questions requiring information from multiple sources",
            "examples": [
                "How does X relate to Y, and what impact does this have on Z?",
                "What is the connection between A and B, and how does it differ from C?"
            ]
        }
    }


    def __init__(self, corpus_path: str, model_name: str = None):
        """
        Initialize QA Generator.

        Args:
            corpus_path: Path to Wikipedia corpus JSON file
            model_name: LLM model name (defaults to config)
        """
        self.logger = get_logger(__name__)
        self.corpus_path = corpus_path
        self.model_name = model_name or LLM_MODEL
        self.corpus = self._load_corpus()

        # Initialize LLM
        llm_factory = LLMFactory()
        self.llm = llm_factory.get_instance(
            model_name=self.model_name,
            task="text-generation",
            max_new_tokens=300,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            is_streaming=False
        )
        self.logger.info(f"QAGenerator initialized with {len(self.corpus)} documents")


    def _load_corpus(self) -> List[Dict]:
        """Load Wikipedia corpus from JSON file."""
        self.logger.info(f"Loading corpus from {self.corpus_path}")
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('pages', [])


    def _sample_documents(self, n: int, strategy: str = "diverse") -> List[Dict]:
        """
        Sample documents from corpus.

        Args:
            n: Number of documents to sample
            strategy: 'diverse' for category-based sampling, 'random' for random sampling

        Returns:
            List of sampled documents
        """
        if strategy == "diverse":
            # Group by category
            by_category = {}
            for doc in self.corpus:
                category = doc.get('category', 'Unknown')
                by_category.setdefault(category, []).append(doc)

            # Sample proportionally from each category
            samples = []
            categories = list(by_category.keys())
            per_category = max(1, n // len(categories))

            for category in categories:
                available = by_category[category]
                sample_size = min(per_category, len(available))
                samples.extend(random.sample(available, sample_size))
                if len(samples) >= n:
                    break

            return samples[:n]
        else:
            return random.sample(self.corpus, min(n, len(self.corpus)))


    def _create_prompt(self, doc: Dict, question_type: str, existing_questions: List[str] = None) -> str:
        """
        Create prompt for LLM to generate questions.

        Args:
            doc: Wikipedia document
            question_type: Type of question to generate
            existing_questions: List of already generated questions to avoid duplicates

        Returns:
            Formatted prompt string
        """
        type_info = self.QUESTION_TYPES[question_type]

        # Extract document content (assuming it's stored in 'extract' or 'content' field)
        content = doc.get('extract', doc.get('content', ''))[:2000]  # Limit to 2000 chars
        title = doc.get('page_title', 'Unknown')

        prompt = f"""Based on the following Wikipedia article, generate a {question_type} question and its answer.

Article Title: {title}
Content: {content}

Question Type: {question_type}
Description: {type_info['description']}
Examples: {', '.join(type_info['examples'][:2])}

Requirements:
1. The question must be answerable from the article content
2. The question should be clear and specific
3. The answer must be accurate and concise (2-3 sentences)
4. For {question_type} questions, follow the style shown in examples

Generate:
Question: <write the question here>
Answer: <write the answer here>
Difficulty: <easy/medium/hard>

Now generate the question and answer:"""

        return prompt


    def _parse_llm_response(self, response: str) -> Dict[str, str]:
        """
        Parse LLM response to extract question, answer, and difficulty.

        Args:
            response: Raw LLM output

        Returns:
            Dictionary with question, answer, and difficulty
        """
        try:
            lines = response.strip().split('\n')
            result = {
                'question': '',
                'answer': '',
                'difficulty': 'medium'
            }

            for line in lines:
                line = line.strip()
                if line.startswith('Question:'):
                    result['question'] = line.replace('Question:', '').strip()
                elif line.startswith('Answer:'):
                    result['answer'] = line.replace('Answer:', '').strip()
                elif line.startswith('Difficulty:'):
                    diff = line.replace('Difficulty:', '').strip().lower()
                    if diff in ['easy', 'medium', 'hard']:
                        result['difficulty'] = diff

            return result
        except Exception as e:
            self.logger.error(f"Error parsing LLM response: {e}")
            return {'question': '', 'answer': '', 'difficulty': 'medium'}


    def generate_qa_pair(self, doc: Dict, question_type: str) -> Dict[str, Any]:
        """
        Generate a single Q&A pair from a document.

        Args:
            doc: Wikipedia document
            question_type: Type of question to generate

        Returns:
            Q&A pair dictionary with metadata
        """
        self.logger.info(f"Generating {question_type} question for: {doc.get('page_title')}")

        # Create prompt
        prompt = self._create_prompt(doc, question_type)

        # Generate using LLM
        try:
            response = self.llm.invoke(prompt)
            parsed = self._parse_llm_response(response)

            # Validate that we got a question and answer
            if not parsed['question'] or not parsed['answer']:
                self.logger.warning(f"Failed to generate valid Q&A for {doc.get('page_title')}")
                return None

            # Create Q&A pair with metadata
            qa_pair = {
                'question': parsed['question'],
                'answer': parsed['answer'],
                'question_type': question_type,
                'difficulty': parsed['difficulty'],
                'source_ids': [doc.get('page_id')],
                'source_urls': [doc.get('url')],
                'source_titles': [doc.get('page_title')],
                'category': doc.get('category'),
                'metadata': {
                    'requires_multiple_docs': question_type == 'multi-hop',
                    'topics': [doc.get('category')]
                }
            }

            return qa_pair

        except Exception as e:
            self.logger.error(f"Error generating Q&A: {e}")
            return None


    def generate_dataset(
        self,
        total_questions: int = 100,
        distribution: Dict[str, int] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate complete Q&A dataset.

        Args:
            total_questions: Total number of Q&A pairs to generate
            distribution: Distribution of question types (defaults to equal distribution)

        Returns:
            List of Q&A pairs
        """
        if distribution is None:
            # Equal distribution across question types
            per_type = total_questions // len(self.QUESTION_TYPES)
            distribution = {qtype: per_type for qtype in self.QUESTION_TYPES.keys()}

        self.logger.info(f"Starting Q&A generation: {total_questions} total")
        self.logger.info(f"Distribution: {distribution}")

        qa_dataset = []
        question_id = 1

        for question_type, count in distribution.items():
            self.logger.info(f"Generating {count} {question_type} questions...")

            # Sample documents for this question type
            sampled_docs = self._sample_documents(count, strategy="diverse")

            for doc in sampled_docs:
                qa_pair = self.generate_qa_pair(doc, question_type)

                if qa_pair:
                    qa_pair['question_id'] = f"Q{question_id:03d}"
                    qa_dataset.append(qa_pair)
                    question_id += 1

                    if len(qa_dataset) % 10 == 0:
                        self.logger.info(f"Generated {len(qa_dataset)}/{total_questions} Q&A pairs")

        self.logger.info(f"Q&A generation complete: {len(qa_dataset)} pairs generated")
        return qa_dataset


    def generate_multi_hop_questions(self, count: int = 25) -> List[Dict[str, Any]]:
        """
        Generate multi-hop questions that require multiple documents.

        Args:
            count: Number of multi-hop questions to generate

        Returns:
            List of multi-hop Q&A pairs
        """
        self.logger.info(f"Generating {count} multi-hop questions...")

        multi_hop_pairs = []
        question_id_start = 1000  # Start from 1000 for multi-hop

        for i in range(count):
            # Sample 2-3 related documents
            num_docs = random.choice([2, 3])
            docs = self._sample_documents(num_docs, strategy="diverse")

            # Combine content from multiple docs
            combined_content = "\n\n".join([
                f"Document {j+1} - {doc.get('page_title')}:\n{doc.get('extract', '')[:800]}"
                for j, doc in enumerate(docs)
            ])

            prompt = f"""Based on the following Wikipedia articles, generate a multi-hop question that requires information from MULTIPLE documents and its answer.

{combined_content}

Requirements:
1. The question MUST require connecting information from at least 2 documents
2. The question should explore relationships, comparisons, or connections
3. The answer must synthesize information from multiple sources
4. Be specific and clear

Generate:
Question: <write the multi-hop question here>
Answer: <write the comprehensive answer here>
Difficulty: <medium/hard>

Now generate the question and answer:"""

            try:
                response = self.llm.invoke(prompt)
                parsed = self._parse_llm_response(response)

                if parsed['question'] and parsed['answer']:
                    qa_pair = {
                        'question_id': f"Q{question_id_start + i:03d}",
                        'question': parsed['question'],
                        'answer': parsed['answer'],
                        'question_type': 'multi-hop',
                        'difficulty': parsed['difficulty'] if parsed['difficulty'] in ['medium', 'hard'] else 'hard',
                        'source_ids': [doc.get('page_id') for doc in docs],
                        'source_urls': [doc.get('url') for doc in docs],
                        'source_titles': [doc.get('page_title') for doc in docs],
                        'category': 'multi-topic',
                        'metadata': {
                            'requires_multiple_docs': True,
                            'num_source_docs': len(docs),
                            'topics': list(set([doc.get('category') for doc in docs]))
                        }
                    }
                    multi_hop_pairs.append(qa_pair)

                    if (i + 1) % 5 == 0:
                        self.logger.info(f"Generated {i + 1}/{count} multi-hop questions")

            except Exception as e:
                self.logger.error(f"Error generating multi-hop question: {e}")
                continue

        self.logger.info(f"Multi-hop generation complete: {len(multi_hop_pairs)} questions")
        return multi_hop_pairs
