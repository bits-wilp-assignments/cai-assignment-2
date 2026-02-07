
import json
from src.util.logging_util import get_logger
from src.wiki.collector import DataCollector
from src.wiki.scrapper import WikiPageScrapper
from src.core.embedding import EmbeddingFactory
from src.core.vector_db import VectorStoreFactory
from src.core.indexer import DataIndexer
from src.config.app_config import *

# Get logger instance
logger = get_logger(__name__)

# Instantiate DataCollector for data collection
collector = DataCollector(
        CATEGORIES,
        FIXED_WIKI_PAGE_FILE,
        RANDOM_WIKI_PAGE_FILE,
        WIKI_USER_AGENT,
        **DATA_COLLECTION_CONFIG
    )

# Instantiate scrapper
scrapper = WikiPageScrapper(
        excluded_section_headers=EXCLUDED_SECTION_HEADERS,
        unwanted_tags_selectors=UNWANTED_TAGS_SELECTORS,
        wiki_user_agent=WIKI_USER_AGENT,
        **TEXT_CHUNKING_CONFIG
    )


def triggr_indexing(is_refresh_fixed: bool = False, is_refresh_random: bool = False):
    logger.info("Initializing indexing process for Hybrid RAG system...")
    logger.info("Starting data collection...")
    collector.collect_data(
        refresh_fixed_wiki_pages=is_refresh_fixed,
        refresh_random_wiki_pages=is_refresh_random,
    )
    logger.info("Data collection completed.")

    wiki_pages = {}
    wiki_documents = []
    with open(FIXED_WIKI_PAGE_FILE, "r", encoding="utf-8") as f:
        wiki_pages = json.load(f)
    for page in wiki_pages['pages'][:10]: # Limiting to first X=10 pages for testing
        logger.info(f"Processing page: {page['page_title']} - {page['url']}")
        documents = scrapper.process_page(
            page['url'],
            page['page_title']
        )
        wiki_documents.extend(documents)
        logger.info(f"Created {len(documents)} documents from page: {page['page_title']}")

    logger.info(f"Total documents created: {len(wiki_documents)}")

     # Create embedding instance
    embeddding_factory = EmbeddingFactory()

    # Get embedding instance for token model
    embedding_instance = embeddding_factory.get_instance(
        model_name=EMBEDDING_MODEL,
    )

    # Create vector store instance
    vector_store = VectorStoreFactory(
        embeddings_instance=embedding_instance,
        data_dir=CHROMA_DB_DIR,
        collection_name=COLLECTION_NAME,
    )
    vector_store_instance = vector_store.get_instance()

    # Create data indexer and index documents
    indexer = DataIndexer(
        vector_store=vector_store_instance,
        collection_name=COLLECTION_NAME,
        db_url=SQL_RECORD_MANAGER_DB_URL,
        bm25_path=BM25_INDEX_PATH
    )
    logger.info("--------------- Starting dense vector indexing... ---------------")
    result = indexer.dense_vector_sync(
        documents=wiki_documents,
        cleanup=CLEAN_UP_STRATEGY,
        doc_metadata_source_id_key=DOC_METADATA_SOURCE_ID_KEY
    )
    logger.info(f"Dense Vector indexing result: {result}")
    logger.info("--------------- Dense vector indexing completed ---------------")
    logger.info("--------------- Starting sparse vector indexing... ---------------")
    sparse_result = indexer.sparse_vector_sync(
        documents=wiki_documents,
        k1=BM25_PARAMS["k1"],
        b=BM25_PARAMS["b"]
    )
    logger.info(f"Sparse Vector indexing result: {sparse_result}")
    logger.info("--------------- Sparse vector indexing completed ---------------")