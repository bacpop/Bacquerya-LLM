import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LANG_DIR = os.path.join(BASE_DIR, "language_models")
DEFAULT_SUMMARISER_LANG_MODEL = os.path.join(LANG_DIR, "flan-t5-xl")
DEFAULT_EMBEDDING_LANG_MODEL = os.path.join(LANG_DIR, "bart-base-finetuned-pubmed")
DEVICE_MAP = 'cpu' ## or cuda