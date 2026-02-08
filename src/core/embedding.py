import torch
from langchain_huggingface import HuggingFaceEmbeddings
from src.util.logging_util import get_logger


class EmbeddingFactory:
    """Singleton factory that manages embedding model instances."""

    _instance = None
    _models = {}  # Dictionary to store model instances by configuration


    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingFactory, cls).__new__(cls)
            cls._instance.logger = get_logger(__name__)
            cls._instance.logger.info("EmbeddingFactory singleton created.")
        return cls._instance


    def get_instance(self, model_name: str, device: str = None, norm_embeddings: bool = False, local_files_only: bool = True) -> HuggingFaceEmbeddings:
        """
        Get or create a singleton embedding model instance for the given configuration.

        Args:
            model_name: HuggingFace model name
            device: Device to use (cuda/mps/cpu), auto-detected if None
            norm_embeddings: Whether to normalize embeddings

        Returns:
            HuggingFaceEmbeddings instance for the given configuration
        """
        # Determine device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        # Create a unique key for this configuration
        config_key = f"{model_name}_{device}_{norm_embeddings}"

        # Return existing instance if available
        if config_key in self._models:
            self.logger.info(f"Returning existing embedding instance for: {model_name} on {device}")
            return self._models[config_key]

        # Create new instance
        self.logger.info(f"Creating new embedding instance for: {model_name} on {device}")
        embedding_instance = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device, "local_files_only": local_files_only},
            encode_kwargs={"normalize_embeddings": norm_embeddings},
        )

        # Store and return
        self._models[config_key] = embedding_instance
        return embedding_instance