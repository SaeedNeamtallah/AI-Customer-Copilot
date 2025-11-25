from abc import ABC, abstractmethod


class LLMInterface(ABC):
    """
    Abstract base class for LLM (Large Language Model) providers.
    
    This interface defines the contract that all LLM providers must implement,
    including text generation and embedding capabilities.
    """

    @abstractmethod
    def set_generation_model(self, model_id: str):
        """Set the model to use for text generation."""
        pass

    @abstractmethod
    def set_embedding_model(self, model_id: str, embedding_size: int):
        """Set the model to use for text embeddings."""
        pass

    @abstractmethod
    def generate_text(
        self,
        prompt: str,
        chat_history: list = [],
        max_output_tokens: int = None,
        temperature: float = None
    ):
        """Generate text based on a prompt and optional chat history."""
        pass

    @abstractmethod
    def embed_text(self, text: str, document_type: str = None):
        """Generate embeddings for the given text."""
        pass

    @abstractmethod
    def construct_prompt(self, prompt: str, role: str):
        """Construct a properly formatted prompt for the LLM provider."""
        pass