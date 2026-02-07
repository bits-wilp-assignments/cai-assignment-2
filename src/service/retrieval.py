from src.util.logging_util import get_logger
from src.core.embedding import EmbeddingFactory
from src.core.vector_db import VectorStoreFactory
from src.core.retriever import HybridRetriever
from src.core.reranking import RerankerFactory
from src.config.app_config import *


# Get logger instance
logger = get_logger(__name__)

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

# Create Reranker
reranker_factory = RerankerFactory()
reranker_instance = reranker_factory.get_instance(
    model_name=RERANKER_MODEL,
)

# Create retriever and perform retrieval
hybrid_retriever = HybridRetriever(
    vector_store=vector_store_instance,
    bm25_path=BM25_INDEX_PATH,
    reranking=IS_RERANKING_ENABLED,
    reranker_model=reranker_instance,
)

def retrieve_contexts(question):
    # Handle dict input from LangChain chains
    if isinstance(question, dict):
        question = question.get("question", question.get("query", ""))
    logger.info(f"Performing retrieval for query: {question}")
    results = hybrid_retriever.retrieve(
        query=question,
        **RETRIEVAL_CONFIG
    )
    logger.info(f"Retrieved {len(results)} documents for query: {question}")
    final_docs = []
    for rank, doc_map in enumerate(results):
        if IS_RETRIEVAL_REPORT_ENABLED:
            # print document metadata
            logger.info(f"Rank {rank + 1}: {doc_map.get('doc').metadata}")
            # print available scores
            score_types = ['dense_score', 'sparse_score', 'rrf_score', 'reranker_score']
            for score_type in score_types:
                if score_type in doc_map:
                    logger.info(f"  {score_type.replace('_', ' ').title()}: {doc_map[score_type]}")
        logger.info("Gathering final document for response.")
        # collect the final documents
        final_docs.append(doc_map['doc'])
    return final_docs