import torch
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from src.util.logging_util import get_logger


class RerankerFactory:
    """Singleton factory that manages cross-encoder reranker model instances."""

    _instance = None
    _rerankers = {}  # Dictionary to store reranker instances by configuration


    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RerankerFactory, cls).__new__(cls)
            cls._instance.logger = get_logger(__name__)
            cls._instance.logger.info("RerankerFactory singleton created.")
        return cls._instance


    def get_instance(self, model_name: str, device: str = None, local_files_only: bool = True) -> HuggingFaceCrossEncoder:
        """
        Get or create a singleton cross-encoder reranker instance for the given configuration.

        Args:
            model_name: HuggingFace cross-encoder model name (e.g., 'cross-encoder/ms-marco-MiniLM-L-6-v2')
            device: Device to use (cuda/mps/cpu), auto-detected if None

        Returns:
            HuggingFaceCrossEncoder instance for the given configuration
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
        config_key = f"{model_name}_{device}"

        # Return existing instance if available
        if config_key in self._rerankers:
            self.logger.info(f"Returning existing reranker instance for: {model_name} on {device}")
            return self._rerankers[config_key]

        # Create new instance
        self.logger.info(f"Creating new reranker instance for: {model_name} on {device}")
        reranker_instance = HuggingFaceCrossEncoder(
            model_name=model_name,
            model_kwargs={"device": device, "local_files_only": local_files_only}
        )

        # Store and return
        self._rerankers[config_key] = reranker_instance
        return reranker_instance
