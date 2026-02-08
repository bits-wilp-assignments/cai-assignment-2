import logging

# -------------------- Logging configuration -------------------
DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DEFAULT_LOG_FILE = None  # e.g. 'logs/app.log'

# ------------------- Model configuration -------------------
MODEL_LOCAL_FILES_ONLY = False  # Set to True to avoid downloading models from HuggingFace
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
IS_NORM_EMBEDDINGS_ENABLED = False
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
# LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"
LLM_MODEL_TASK = "text-generation"
LLM_CONFIG = {
    "max_new_tokens": 500,
    "temperature": 0.001,
    "repetition_penalty": 1.0,
    "top_p": 0.95,
    "torch_dtype": "auto",
    "device_map": "auto",
    "do_sample": True,
    "is_streaming": True,
}

# ------------------- Indexer configuration -------------------
CHROMA_DB_DIR = "./data/chroma_db"
COLLECTION_NAME = "wikipedia_collection"
SQL_RECORD_MANAGER_DB_URL = "sqlite:///data/chroma_db/record_manager.sql"
DOC_METADATA_SOURCE_ID_KEY = "source"
CLEAN_UP_STRATEGY = "incremental"  # Options: 'full', 'incremental', 'none'
BM25_INDEX_PATH = "./data/bm25_index"
BM25_PARAMS = {
    "k1": 1.5,
    "b": 0.75,
}

# ------------------- Retriever configuration -------------------
IS_RERANKING_ENABLED = True
RETRIEVAL_CONFIG = {
    "dense_top_k": 5,
    "sparse_top_k": 5,
    "rrf_top_k": 4,
    "reranker_top_k": 2,
    "rrf_k": 60,
}
IS_RETRIEVAL_REPORT_ENABLED = True

# ------------------- Wikipedia configuration -------------------
WIKI_USER_AGENT = "HybridRAGBot/1.0 (2024aa05193@wilp.bits-pilani.ac.in)"

# Predefined categories for fixed wiki pages
CATEGORIES = [
    "Quantum_mechanics",
    "Renaissance_art",
    "Classical_music",
    "Astronomy",
    "Philosophy",
    "Mythology",
    "Computer_science",
    "Environmental_science",
    "Ancient_history",
    "Political_science",
    "Geography",
    "Literature",
]

# File paths for storing wiki page data
FIXED_WIKI_PAGE_FILE = "data/corpus/fixed_wiki_pages.json"
RANDOM_WIKI_PAGE_FILE = "data/corpus/random_wiki_pages.json"

# Configuration for data collection limits
DATA_COLLECTION_CONFIG = {
    "fixed_sample_size": 200,
    "fixed_max_pages": 220,
    "fixed_max_cat_pages": 20,
    "fixed_min_word_count": 200,
    "random_sample_size": 300,
    "random_max_pages": 400,
    "random_min_page_size": 8000,
    "random_seed": 42,
}

# List of tags and classes to remove from wiki pages during pre-processing
UNWANTED_TAGS_SELECTORS = [
    "sup.reference",
    "table",              # Exclude tables duiring initial content extraction
    "style",
    "script",
    "nav",
    "footer",
    "img",                # Exclude images during initial content extraction
    ".toc",
    ".mw-editsection",
    ".gallery",           # Exclude galleries during initial content extraction
    "figcaption",         # Exclude figure captions during initial content extraction
    ".infobox",
    ".box-Notability",
    ".metadata"
]

# List of section headers to exclude during content extraction
EXCLUDED_SECTION_HEADERS = [
    "See also",
    "References",
    "External links",
    "Further reading",
    "Notes"
]

# Configuration for text chunking
TEXT_CHUNKING_CONFIG = {
    "max_tokens": 400,
    "overlap_sents": 2,
}