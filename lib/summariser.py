from transformers import pipeline
import os
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
        return self.summariser(text,
            max_length=max_length,
            min_length=min_length,
            do_sample=do_sample,
            temperature=temperature
            )

    def get_summariser(self):
        ##TODO add local and remote model saving/loading
        self.summariser = pipeline(
            "summarization",
            self.summariser_model_path,
            device_map=self.device_map
        )

    def get_sentence_embedding_model(self):
        ##TODO add local and remote model saving/loading
        self.embedding_model = SentenceTransformer(self.embedding_model_path, device=self.device_map)

    def get_query_embedding(self, query:str):
        self.emb_query = self.embedding_model.encode(query, convert_to_tensor=True)

    def get_similarity_score(self, text:str):
        """
        outputs cosine similarity score between the query (gene and species) and the text
        :param text:
        :return:
        """
        emb_texts = self.embedding_model.encode(text, convert_to_tensor=True)
        return self.emb_query @ emb_texts.T

#from transformers import T5Tokenizer, T5ForConditionalGeneration
#tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
#model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large", device_map=0)