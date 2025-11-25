"""
Cohere Provider Module

This module implements the LLM interface for Cohere's API, providing
text generation and embedding capabilities using Cohere models.
"""

from ..LLMInterface import LLMInterface
from ..LLMEnums import CoHereEnums, DocumentTypeEnum
import cohere
import logging


class CoHereProvider(LLMInterface):
    """
    Cohere API Provider Implementation.
    
    This class provides an interface to Cohere's language models for both
    text generation (chat) and text embeddings.
    """
    def __init__(
        self,
        api_key: str,
        default_input_max_characters: int = 1000,
        default_generation_max_output_tokens: int = 1000,
        default_generation_temperature: float = 0.1,
    ):
        """
        Initialize the Cohere Provider.
        
        Args:
            api_key (str): Cohere API key for authentication
            default_input_max_characters (int): Maximum characters to process from input text
            default_generation_max_output_tokens (int): Default max tokens for text generation
            default_generation_temperature (float): Default temperature for generation (0.0-5.0)
        """
        # API configuration
        self.api_key = api_key
        self.client = cohere.Client(api_key=self.api_key)

        # Default parameters for text processing and generation
        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature

        # Model configuration
        self.generation_model_id = None  # Set via set_generation_model()
        self.embedding_model_id = None   # Set via set_embedding_model()
        self.embedding_size = None       # Embedding vector embedding_size

        self.enums = CoHereEnums

        # Logger instance
        self.logger = logging.getLogger(__name__)

    def set_generation_model(self, model_id: str):
        """
        Set the model to use for text generation.
        
        Args:
            model_id (str): Cohere model identifier (e.g., 'command', 'command-light')
        """
        self.generation_model_id = model_id

    def set_embedding_model(self, model_id: str, embedding_size: int):
        """
        Set the model to use for text embeddings.
        
        Args:
            model_id (str): Cohere embedding model identifier (e.g., 'embed-english-v3.0')
            embedding_size (int): embedding_size of the embedding vectors produced by the model
        """
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size

    def process_text(self, text: str):
        """
        Clean and truncate text to the maximum allowed character limit.
        
        Args:
            text (str): The input text to process
            
        Returns:
            str: Cleaned and truncated text limited to default_input_max_characters
        """
        return text[: self.default_input_max_characters].strip()

    def generate_text(
        self,
        prompt: str,
        chat_history: list = [],
        max_output_tokens: int = None,
        temperature: float = None,
    ):
        """
        Generate text using Cohere's chat API.
        
        Args:
            prompt (str): The user prompt/message to generate text for
            chat_history (list): List of previous messages in the conversation
            max_output_tokens (int, optional): Maximum tokens in the generated response
            temperature (float, optional): Sampling temperature (0.0 = deterministic, 5.0 = creative)
            
        Returns:
            str: Generated text response from the model, or None if an error occurs
            
        Raises:
            None: Errors are logged and None is returned
        """
        # Validate that the client is initialized
        if not self.client:
            self.logger.error("CoHere client was not set")
            return None

        # Validate that a generation model has been configured
        if not self.generation_model_id:
            self.logger.error("Generation model for CoHere was not set")
            return None

        # Use provided parameters or fall back to defaults
        max_output_tokens = (
            max_output_tokens
            if max_output_tokens
            else self.default_generation_max_output_tokens
        )
        temperature = (
            temperature if temperature else self.default_generation_temperature
        )

        # Call Cohere API for chat completion
        response = self.client.chat(
            model=self.generation_model_id,
            chat_history=chat_history,
            message=self.process_text(prompt),
            temperature=temperature,
            max_tokens=max_output_tokens,
        )

        # Validate response
        if not response or not response.text:
            self.logger.error("Error while generating text with CoHere")
            return None

        return response.text

    def embed_text(self, text: str, document_type: str = None):
        """
        Generate embeddings for the given text using Cohere's embedding API.
        
        Args:
            text (str): The text to generate embeddings for
            document_type (str, optional): Type of document - 'query' or 'document'
                                          Affects the input_type parameter for Cohere API
            
        Returns:
            list: Embedding vector as a list of floats, or None if an error occurs
            
        Raises:
            None: Errors are logged and None is returned
        """
        # Validate that the client is initialized
        if not self.client:
            self.logger.error("CoHere client was not set")
            return None

        # Validate that an embedding model has been configured
        if not self.embedding_model_id:
            self.logger.error("Embedding model for CoHere was not set")
            return None

        # Determine input type based on document_type
        input_type = CoHereEnums.DOCUMENT.value
        if document_type == DocumentTypeEnum.QUERY.value:
            input_type = CoHereEnums.QUERY.value

        # Call Cohere API to generate embeddings
        response = self.client.embed(
            model=self.embedding_model_id,
            texts=[self.process_text(text)],
            input_type=input_type,
            embedding_types=["float"],
        )

        # Validate response structure
        if not response or not response.embeddings or not response.embeddings.float:
            self.logger.error("Error while embedding text with CoHere")
            return None

        return response.embeddings.float[0]

    def construct_prompt(self, prompt: str, role: str):
        """
        Construct a message object for Cohere's chat API.
        
        Args:
            prompt (str): The message content
            role (str): The role of the message sender (e.g., 'USER', 'CHATBOT', 'SYSTEM')
            
        Returns:
            dict: Message object with 'role' and 'text' keys
        """
        return {"role": role, "text": self.process_text(prompt)}
