
import hashlib
import os
import shutil
import bm25s
from langchain_classic.indexes import index, SQLRecordManager
import Stemmer
from src.util.logging_util import get_logger


class DataIndexer:

    def __init__(self, vector_store, collection_name: str, db_url: str, bm25_path: str):
        self.logger = get_logger(__name__)
        self.vector_store = vector_store
        self.collection_name = collection_name
        self.db_url = db_url
        self.record_manager = SQLRecordManager(
            namespace=f"chroma/{self.collection_name}",
            db_url=self.db_url
        )
        self.record_manager.create_schema()
        self.bm25_path = bm25_path
        self.logger.info("DataIndexer initialized.")

    def dense_vector_sync(self, documents, cleanup="incremental", doc_metadata_source_id_key="source"):
        self.logger.info(f"Indexing {len(documents)} documents into collection: {self.collection_name}.")
        result = index(
            docs_source=documents,
            record_manager=self.record_manager,
            vector_store=self.vector_store,
            cleanup=cleanup,
            source_id_key=doc_metadata_source_id_key,
            key_encoder=lambda doc: hashlib.sha256(doc.page_content.encode()).hexdigest()
        )
        self.logger.info("Indexing completed.")
        return result


    def sparse_vector_sync(self, documents, k1=1.5, b=0.75):
        self.logger.info(f"Creating BM25 index for {len(documents)} documents.")
        try:
            # Tokenize using the faster bm25s built-in tokenizer
            corpus_tokens = bm25s.tokenize(
                [doc.page_content for doc in documents],
                stopwords="en",
                stemmer=Stemmer.Stemmer("english")
            )

            dict_corpus = [
                {"text": doc.page_content, "metadata": doc.metadata}
                for doc in documents
            ]

            # Create and save the index
            retriever = bm25s.BM25(k1, b)
            retriever.index(corpus_tokens)
            # Remove existing index directory if it exists
            if os.path.exists(self.bm25_path):
                shutil.rmtree(self.bm25_path)
            retriever.save(self.bm25_path, corpus=dict_corpus)
        except Exception as e:
            self.logger.error(f"Error creating BM25 index: {e}")
            return {"status": "error", "message": str(e)}
        self.logger.info("BM25 index created and saved.")
        return {"status": "BM25 index created", "num_documents": len(documents)}