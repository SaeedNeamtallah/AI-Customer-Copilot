from openai import OpenAI
from stores.llm.LLMInterface import LLMInterface
import logging
from ..LLMEnums import OpenAIEnums
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenAIProvider(LLMInterface):
    def __init__(
        self,
        api_key: str,
        base_url: str = None,
        default_input_max_characters: int = 1000,
        default_generation_max_output_tokens: int = 1000,
        default_generation_temperature: float = 0.1,
    ):
        self.api_key = api_key
        self.base_url = base_url if base_url and base_url.strip() else None
        self.client = OpenAI(api_key=api_key, base_url=self.base_url)
        self.generation_model = None
        self.embedding_model = None
        self.embedding_size = None
        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature
        self.logger = logging.getLogger(__name__)
        self.enums = OpenAIEnums
        
    def set_generation_model(self, model_id: str):
        self.generation_model = model_id

    def set_embedding_model(self, model_id: str, embedding_size: int):
        self.embedding_model = model_id
        self.embedding_size = embedding_size

    def generate_text(
        self,
        prompt: str,
        chat_history: list = [],
        max_output_tokens: int = None,
        temperature: float = None,
    ):
        if not self.generation_model:
            raise ValueError("Generation model is not set.")
        max_output_tokens = (
            max_output_tokens or self.default_generation_max_output_tokens
        )
        temperature = temperature or self.default_generation_temperature
        
        user_message = self.construct_prompt(prompt, role=OpenAIEnums.USER.value)
        messages = chat_history + [user_message]

        response = self.client.chat.completions.create(
            model=self.generation_model,
            messages=messages,
            max_tokens=max_output_tokens,
            temperature=temperature,
        )       
        
        if (
            not response
            or not response.choices
            or len(response.choices) == 0
            or not response.choices[0].message
        ):
            self.logger.error("Error while generating text with OpenAI")
            return None

        return self.process_text_response(response)

    def process_text_response(self, response):
        return response.choices[0].message.content

    def embed_text(self, text: str, document_type: str = None):
        if not self.embedding_model:
            raise ValueError("Embedding model is not set.")

        response = self.client.embeddings.create(input=text, model=self.embedding_model)

        if (
            not response
            or not response.data
            or len(response.data) == 0
            or not response.data[0].embedding
        ):
            self.logger.error("Error while embedding text with OpenAI")
            return None

        return self.process_embedding_response(response)

    def process_embedding_response(self, response):
        return response.data[0].embedding

    def construct_prompt(self, prompt: str, role: str):
        return {"role": role, "content": self.process_text(prompt)}

    def process_text(self, text: str):
        text = text.strip()
        return text[: self.default_input_max_characters]
