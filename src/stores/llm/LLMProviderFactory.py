"""
LLM Provider Factory Module

This module provides a factory pattern implementation for creating
LLM provider instances based on configuration.
"""

from .LLMEnums import LLMEnums
from .providers import CoHereProvider, OpenAIProvider


class LLMProviderFactory:

    def __init__(self, config: dict):
        self.config = config

    def get_provider(self, provider_name: str):
        """
        Get an LLM provider instance based on the provider name.
        
        Args:
            provider_name (str): Name of the LLM provider (e.g., 'OPENAI', 'COHERE')
            
        Returns:
            LLMInterface: An instance of the requested LLM provider
            
        Raises:
            ValueError: If the requested provider is not supported
        """
        # Create Cohere provider instance
        if provider_name == LLMEnums.COHERE.value:
            return CoHereProvider(
                api_key=self.config.COHERE_API_KEY,
                default_input_max_characters=self.config.INPUT_DEFAULT_MAX_CHARACTERS,
                default_generation_max_output_tokens=self.config.GENERATION_DEFAULT_MAX_TOKENS,
                default_generation_temperature=self.config.GENERATION_DEFAULT_TEMPERATURE
            )

        # Create OpenAI provider instance
        elif provider_name == LLMEnums.OPENAI.value:
            return OpenAIProvider(
                api_key=self.config.OPENAI_API_KEY,
                base_url=self.config.OPENAI_API_URL,
                default_input_max_characters=self.config.INPUT_DEFAULT_MAX_CHARACTERS,
                default_generation_max_output_tokens=self.config.GENERATION_DEFAULT_MAX_TOKENS,
                default_generation_temperature=self.config.GENERATION_DEFAULT_TEMPERATURE
            )

        # Unsupported provider
        else:
            raise ValueError(f"Unsupported LLM provider: {provider_name}")