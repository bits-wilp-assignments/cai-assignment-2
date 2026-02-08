import os
import bm25s
import Stemmer
import hashlib
from langchain_classic.schema import Document
from src.util.logging_util import get_logger


class HybridRetriever:

    def __init__(self, vector_store, bm25_path, reranking = False, reranker_model = None):
        self.logger = get_logger(__name__)
        self.vector_store = vector_store
        self.bm25_path = bm25_path
        self._bm25_index = None  # Lazy loading - load only when needed
        self.reranking = reranking
        if self.reranking:
            if reranker_model is None:
                raise ValueError("Reranker model must be provided if reranking is enabled.")
            self.reranker_model = reranker_model
        self.logger.info("HybridRetriever initialized with vector store (BM25 will be loaded on-demand).")

    @property
    def bm25_index(self):
        """Lazy load BM25 index only when accessed."""
        if self._bm25_index is None:
            if self.bm25_exists():
                self.logger.info(f"Loading BM25 index from: {self.bm25_path}")
                self._bm25_index = bm25s.BM25.load(self.bm25_path, load_corpus=True, mmap=True)
                self.logger.info("BM25 index loaded successfully.")
            else:
                self.logger.warning(f"BM25 index not found at: {self.bm25_path}. Sparse retrieval will be skipped.")
        return self._bm25_index

    def bm25_exists(self) -> bool:
        """Check if BM25 index files exist."""
        if not os.path.exists(self.bm25_path):
            return False
        # Check for essential BM25 index files
        required_files = ['params.index.json', 'vocab.index.json']
        return all(os.path.exists(os.path.join(self.bm25_path, f)) for f in required_files)

    def reload_bm25_index(self):
        """Force reload of BM25 index. Useful after indexing is complete."""
        self.logger.info("Reloading BM25 index...")
        self._bm25_index = None  # Clear cached index
        _ = self.bm25_index  # Trigger lazy loading
        if self._bm25_index is not None:
            self.logger.info("BM25 index reloaded successfully.")
            return True
        else:
            self.logger.warning("BM25 index reload failed - index files not found.")
            return False

    def _get_doc_id(self, content: str) -> str:
        normalized = "".join(content.split()).lower()
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


    def _add_to_rrf(self, docs, scores, score_key, doc_map, rrf_k):
        for rank, doc in enumerate(docs):
            doc_id = self._get_doc_id(doc.page_content)
            if doc_id not in doc_map:
                doc_map[doc_id] = {"doc": doc, score_key: scores[rank]}
            else:
                doc_map[doc_id][score_key] = scores[rank]
            doc_map[doc_id]["rrf_score"] = doc_map[doc_id].get("rrf_score", 0.0) + 1.0 / (rrf_k + (rank+1))


    def retrieve(self, query, dense_top_k=15, sparse_top_k=15, rrf_top_k=10, reranker_top_k=8, rrf_k=60):
        if isinstance(query, dict):
            query = query.get("question", query.get("query", ""))
        self.logger.info(f"Retrieving top {dense_top_k} documents for the query.")
        # DENSE RETRIEVAL: Retrieve from vector store
        results_with_scores = self.vector_store.similarity_search_with_score(query, k=dense_top_k)
        dense_docs, dense_scores = map(list, zip(*results_with_scores))
        self.logger.debug(f"Dense documents retrieved: {[doc.metadata for doc in dense_docs]}")
        self.logger.debug(f"Dense scores: {dense_scores}")
        # dense_ranks = np.argsort(np.array(dense_scores))
        # self.logger.debug(f"Dense ranks: {dense_ranks}")

        # SPARSE RETRIEVAL: Retrieve from BM25 (if available)
        sparse_docs = []
        sparse_scores = []
        if self.bm25_index is not None:
            query_tokens = bm25s.tokenize(
                    [query],
                    stopwords="en",
                    stemmer=Stemmer.Stemmer("english")
                )
            sparse_results, sparse_scores = self.bm25_index.retrieve(query_tokens, k=sparse_top_k)
            sparse_docs = [Document(page_content=doc["text"], metadata=doc["metadata"]) for doc in sparse_results[0]]
            sparse_scores = sparse_scores[0]  # Extract scores for the single query
            self.logger.debug(f"Sparse documents retrieved: {[doc.metadata for doc in sparse_docs]}")
            self.logger.debug(f"Sparse scores: {sparse_scores}")
        else:
            self.logger.info("BM25 index not available - using dense retrieval only")
        # sparse_ranks = np.argsort(-sparse_scores)
        # self.logger.debug(f"Sparse ranks: {sparse_ranks}")

        # RRF: Combine results using Reciprocal Rank Fusion
        doc_map = {}
        # Add dense and sparse rankings
        self._add_to_rrf(dense_docs, dense_scores, "dense_score", doc_map, rrf_k)
        if sparse_docs:  # Only add sparse results if available
            self._add_to_rrf(sparse_docs, sparse_scores, "sparse_score", doc_map, rrf_k)

        # Sort doc_map by RRF scores
        final_ranked_docs = sorted(
            doc_map.values(),
            key=lambda x: x["rrf_score"],
            reverse=True
        )[:rrf_top_k]

        # RERANKING: Apply reranker if enabled
        if self.reranking:
            self.logger.info(f"Reranking top {reranker_top_k} documents using the reranker model.")
            reranker_inputs = [
                (query, doc_map["doc"].page_content) for doc_map in final_ranked_docs
            ]
            reranker_scores = self.reranker_model.score(reranker_inputs)
            # Update doc_map with reranker scores
            for idx, doc_map in enumerate(final_ranked_docs):
                doc_map["reranker_score"] = reranker_scores[idx]
            # Sort by reranker scores
            final_ranked_docs = sorted(
                final_ranked_docs,
                key=lambda x: x["reranker_score"],
                reverse=True
            )[:reranker_top_k]

        return final_ranked_docs

