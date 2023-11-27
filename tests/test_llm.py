from lib.llm import LLM
import unittest
from config import LANG_DIR, DEVICE_MAP
import os


@unittest.skipIf(not os.path.exists(LANG_DIR) or len(os.listdir(LANG_DIR)) == 0, "No Large Language Model is available")
class TestLLM(unittest.TestCase):
    def setUp(self):
        self.llm = LLM(
            main_model_path=os.path.join(LANG_DIR, "sciphi-mistral-7b-32k.Q5_K_M.gguf"), ## Maybe add auto select a model, instead of hard-coded models
            embedding_model_path=os.path.join(LANG_DIR, "bart-base-finetuned-pubmed"),
            device_map=DEVICE_MAP,
            model_type='gguf',
            chat_type='llama'
        )
        self.llm.load_main_LLM()
        self.llm.load_embedding_model()

    def test_get_sentence_embedding(self):
        self.llm.get_sentence_embedding("This is a test")

    def test_chat(self):
        self.llm.chat("Why the sky is blue?")


