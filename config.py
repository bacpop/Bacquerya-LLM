import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LANG_DIR = os.path.join(BASE_DIR, "language_models")
DEFAULT_SUMMARISER_LANG_MODEL = os.path.join(LANG_DIR, "sciphi-mistral-7b-32k.Q5_K_M.gguf")
DEFAULT_EMBEDDING_LANG_MODEL = os.path.join(LANG_DIR, "bart-base-finetuned-pubmed")
LLM_SYSTEM_PROMPT="You are a scientist experienced in the field of biology. You give very very short, precise and abstractive summaries of given papers or paragraphs."
#DEFAULT_EMBEDDING_LANG_MODEL = os.path.join(LANG_DIR, "bge-large-en-v1.5")
DEVICE_MAP = 'cpu' ## or cuda