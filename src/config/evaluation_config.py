# ------------------- Storage Configuration -------------------
# Directory for storing Q&A datasets and evaluation results
EVALUATION_STORAGE_DIR = "./data/evaluation"

# Default filename for Q&A dataset (without timestamp)
DEFAULT_QA_DATASET_FILENAME = "qa_dataset.json"

# ------------------- Q&A Generation Configuration -------------------
# LLM parameters for Q&A generation
QA_GENERATION_CONFIG = {
    "max_new_tokens": 300,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
    "is_streaming": False,
}

# Question type definitions and examples
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

# Default question distribution for dataset generation
DEFAULT_QUESTION_DISTRIBUTION = {
    'factual': 30,
    'comparative': 25,
    'inferential': 25,
    'multi-hop': 20
}

# Default total questions for generation
DEFAULT_TOTAL_QUESTIONS = 100

# ------------------- Evaluation Configuration -------------------
# Default retrieval top-k values for evaluation
EVALUATION_RETRIEVAL_TOP_K = 10

# Precision/Recall/Hit Rate k values to calculate
EVALUATION_K_VALUES = [3, 5, 10]

# Default corpus path for Q&A generation
DEFAULT_CORPUS_PATH = "./data/fixed_wiki_pages.json"

# ------------------- Q&A Validation Configuration -------------------
# Required fields for Q&A pairs
QA_REQUIRED_FIELDS = [
    'question_id',
    'question',
    'answer',
    'question_type',
    'source_ids',
    'source_urls'
]

# Optional fields for Q&A pairs
QA_OPTIONAL_FIELDS = [
    'source_titles',
    'category',
    'difficulty',
    'metadata'
]

# Valid question types (must match QUESTION_TYPES keys)
VALID_QUESTION_TYPES = ['factual', 'comparative', 'inferential', 'multi-hop']

# Valid difficulty levels
VALID_DIFFICULTIES = ['easy', 'medium', 'hard']

# Placeholder patterns to detect and reject during QA generation
QA_PLACEHOLDER_PATTERNS = [
    '<write',
    'write the question here',
    'write the answer here',
    'write your',
    '[insert',
    'TODO',
    'TBD'
]

# Minimum length for valid questions and answers
MIN_QUESTION_LENGTH = 10
MIN_ANSWER_LENGTH = 20

# ------------------- Answer Quality Evaluation Configuration -------------------
# Enable answer quality evaluation by default
ENABLE_ANSWER_EVALUATION = True

# Maximum number of questions to evaluate for answer quality (None = all)
DEFAULT_MAX_ANSWER_EVALUATIONS = None

# ------------------- Retrieval Quality Evaluation Configuration -------------------
# Maximum number of questions to evaluate for retrieval quality (None = all)
DEFAULT_MAX_RETRIEVAL_EVALUATIONS = None
