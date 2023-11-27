from transformers import pipeline
from sentence_transformers import SentenceTransformer
from typing import Literal


class LLM:
    """
    Base class for LLMs
    Loads the summariser and the sentence embedding model
    """
    def __init__(self, main_model_path, embedding_model_path, device_map='auto', model_type=None, chat_type=None):
        self.main_model_path = main_model_path
        self.embedding_model_path = embedding_model_path
        self.device_map = device_map
        self.model_type = model_type
        self.chat_type = chat_type
        self.main_model = None
        self.embedding_model = None

    def load_main_LLM(self, n_ctx=8000, task:Literal["summarization", "text-classification", "automatic-speech-recognition"]="summarization"):

        if self.model_type is None:
            self.main_model = pipeline(
                task,
                self.main_model_path,
                device_map=self.device_map
            )
            return self.main_model

        elif self.model_type.lower() == "gguf":
            ##Load Llama model
            from llama_cpp import Llama
            self.main_model = Llama(self.main_model_path, n_ctx=n_ctx)
            return self.main_model

    def load_embedding_model(self):
        self.embedding_model = SentenceTransformer(self.embedding_model_path, device=self.device_map)

    def get_sentence_embedding(self, text:str):
        return self.embedding_model.encode(text, convert_to_tensor=True)

    @staticmethod
    def _prepare_prompt4llama(instruction_prompt=None, system_prompt=None):
        from config import LLM_SYSTEM_PROMPT
        system_prompt = system_prompt if system_prompt is not None else LLM_SYSTEM_PROMPT

        prompt = f"""
            ### System:
            {system_prompt}

            ### Instruction:
            {instruction_prompt}

            ### Response:
            ...
            """
        return prompt

    def chat(self, instruction_prompt, max_length=150, min_length=100, do_sample=None, temperature=0.2,
             system_prompt=None):
        if self.chat_type.lower() == "llama":
            prompt = self._prepare_prompt4llama(instruction_prompt, system_prompt)
            return self.main_model(prompt, max_tokens=max_length * 8, temperature=temperature)['choices'][0][
                'text'].replace("\n", " ").lstrip()
        else:
            return self.main_model(instruction_prompt, max_length=max_length, min_length=min_length, temperature=temperature, do_sample=do_sample)