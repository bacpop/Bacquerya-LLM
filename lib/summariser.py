from transformers import pipeline
from sentence_transformers import SentenceTransformer


class Summariser:
    def __init__(self, summariser_model_path, embedding_model_path, device_map='auto'):
        self.summariser_model_path = summariser_model_path
        self.embedding_model_path = embedding_model_path
        self.device_map = device_map
        self.summariser = None
        self.embedding_model = None
        self.emb_query = None

    def load_models(self):
        self.get_summariser()
        self.get_sentence_embedding_model()

    def summarise(self, text, max_length=150, min_length=100, do_sample=None, temperature=0.2):
        '''
        Main method to summarise a given text.
        :param text: input text
        :param max_length: of words in the summary
        :param min_length:
        :param do_sample: somewhat related to the randomness of the summariser
        :param temperature: creativity of the summariser
        :return: summary text
        '''
        if self.summariser_model_path.lower().endswith(".gguf"):
            prompt = self._prepare_prompt4llama(text, max_length)
            return self.summariser(prompt, max_tokens=max_length*8,temperature=temperature)['choices'][0]['text'].replace("\n", " ").lstrip()

        return self.summariser(text,
            max_length=max_length,
            min_length=min_length,
            do_sample=do_sample,
            temperature=temperature
            )

    def get_summariser(self):
        if self.summariser_model_path.lower().endswith(".gguf"):
            ##Load Llama model
            from llama_cpp import Llama
            self.summariser = Llama(self.summariser_model_path)
            return self.summariser
        self.summariser = pipeline(
            "summarization",
            self.summariser_model_path,
            device_map=self.device_map
        )
        return self.summariser

    def get_sentence_embedding_model(self):
        self.embedding_model = SentenceTransformer(self.embedding_model_path, device=self.device_map)

    def get_query_embedding(self, query:str):
        self.emb_query = self.embedding_model.encode(query, convert_to_tensor=True)

    def get_similarity_score(self, text:str):
        """
        outputs cosine similarity score between the query (gene and species) and the text
        :param text:
        :return:
        """
        from torch import cosine_similarity
        emb_text = self.embedding_model.encode(text, convert_to_tensor=True)

        return cosine_similarity(self.emb_query, emb_text, dim=0)

    @staticmethod
    def _prepare_prompt4llama(text, max_length):
        from config import LLM_SYSTEM_PROMPT
        prompt = f"""
        ### System:
        {LLM_SYSTEM_PROMPT}

        ### Instruction:
        Summarise the following text in {max_length//2} words: '{text}'

        ### Response:
        ...
        """
        return prompt

#from transformers import T5Tokenizer, T5ForConditionalGeneration
#tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
#model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large", device_map=0)