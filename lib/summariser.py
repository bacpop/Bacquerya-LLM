from lib.llm import LLM
from config import LLM_SUMMARISER_SYSTEM_PROMPT
import logging
import os
import json


class Summariser:
    def __init__(self, language_model:LLM, instruction_prompt=None, system_prompt=None, **kwargs):
        self.language_model = language_model
        self.summariser = language_model.main_model
        self.embedding_model = language_model.embedding_model
        self.max_length = kwargs.get("max_length", 100)
        self.min_length = kwargs.get("min_length", 20)
        self.do_sample = kwargs.get("do_sample", False)
        self.temperature = kwargs.get("temperature", 0.2)
        self.query = kwargs.get("query")
        self.texts = kwargs.get("texts")
        self.similarity_threshold = kwargs.get("similarity_threshold", 0.0)
        self.re_summarise = kwargs.get("re_summarise", True)
        self.combine_summaries = kwargs.get("combine_summaries", True)

        self.instruction_prompt = instruction_prompt
        self.system_prompt = system_prompt if system_prompt is not None else LLM_SUMMARISER_SYSTEM_PROMPT

        self.emb_query = None

    def process(self):
        ##TODO: divide this process into sub-processes, maybe NextFlow?
        self.get_query_embedding(self.query) if self.embedding_model is not None else None

        if os.path.isfile(self.texts) and self.texts.endswith(".json"):
            with open(self.texts, "r") as f:
                self.texts = json.load(f)

        output_lst = []
        for text in self.texts:
            text_summary = self.summarise(text)

            if self.re_summarise:
                text_summary = self.summarise(text=text_summary)

            output_lst.append(
                {#"text": text,  ## later remove this, this only stays for development
                 "summary": text_summary,
                 "query_similarity_score": self.get_query_similarity_score(text_summary),
                 "summary_similarity_score": self.get_summary_similarity_score(text, text_summary) if self.embedding_model is not None else None,
                 })

        if self.combine_summaries:
            output_lst.append(self.get_final_summary(output_lst))

        return output_lst

    def summarise(self, text):
        instruction_prompt = self.instruction_prompt if self.instruction_prompt is not None else f"Summarise the following text under {self.max_length} words: {text}"
        return self.language_model.chat(
            instruction_prompt=instruction_prompt,
            max_length=self.max_length,
            min_length=self.min_length,
            do_sample=self.do_sample,
            temperature=self.temperature,
            system_prompt=self.system_prompt
        )

    def get_query_embedding(self, query:str):
        self.emb_query = self.embedding_model.encode(query, convert_to_tensor=True) if self.embedding_model is not None else None
        return self.emb_query

    def get_query_similarity_score(self, text:str):
        """
        outputs cosine similarity score between the query and the summary
        :param text:
        :return:
        """
        from torch import cosine_similarity
        if self.embedding_model is None: return None
        emb_text = self.embedding_model.encode(text, convert_to_tensor=True)

        return float(cosine_similarity(self.emb_query, emb_text, dim=0))

    def get_summary_similarity_score(self, text:str, summary:str):
        """
        outputs cosine similarity score between the text and the summary
        :param text:
        :return:
        """
        from torch import cosine_similarity
        if self.embedding_model is None: return None
        emb_text = self.embedding_model.encode(text, convert_to_tensor=True)
        emb_summary = self.embedding_model.encode(summary, convert_to_tensor=True)

        return float(cosine_similarity(emb_text, emb_summary, dim=0))

    def get_final_summary(self, summaries:list, similarity_threshold:float=0.0):
        """
        Combine summaries that are similar by query threshold
        :param summaries:
        :param similarity_threshold:
        :return:
        """
        combined_summaries_text = ' '.join([summary["summary"] for summary in summaries if summary["query_similarity_score"] >= similarity_threshold])
        logging.info(combined_summaries_text)
        final_summary = self.summarise(combined_summaries_text)
        logging.info(final_summary)

        return {
            #"text": combined_summaries_text,
            "summary": final_summary,
            "query_similarity_score": self.get_query_similarity_score(final_summary),
            "summary_similarity_score": self.get_summary_similarity_score(combined_summaries_text, final_summary),
             }