from langchain_chroma import Chroma
from src.util.logging_util import get_logger


class VectorStoreFactory:

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(VectorStoreFactory, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance


    def __init__(self, embeddings_instance, data_dir: str, collection_name: str):
        # Avoid re-initialization
        if self._initialized:
            return
        self.logger = get_logger(__name__)
        self.embeddings_instance = embeddings_instance
        self.data_dir = data_dir
        self.collection_name = collection_name
        # Create the Chroma vector store instance
        self.vector_store = Chroma(
            persist_directory=self.data_dir,
            embedding_function=self.embeddings_instance,
            collection_name=self.collection_name,
            collection_metadata={"hnsw:space": "cosine"}
        )
        self._initialized = True
        self.logger.info("Singleton - VectorStoreFactory initialized.")


    def get_instance(self) -> Chroma:
        self.logger.info("Providing Chroma vector store instance.")
        return self.vector_store