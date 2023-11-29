import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LANG_DIR = os.path.join(BASE_DIR, "language_models")

TEST_DIR = os.path.join(BASE_DIR, 'tests')
TEST_DATA_DIR = os.path.join(TEST_DIR, 'test_data')


SUMMARISER_LANG_MODEL = os.path.join(LANG_DIR, "sciphi-mistral-7b-32k.Q5_K_M.gguf")
EMBEDDING_LANG_MODEL = None  #os.path.join(LANG_DIR, "bart-base-finetuned-pubmed")
LLM_SUMMARISER_SYSTEM_PROMPT="You are a scientist experienced in the field of biology. You give very very short, precise and abstractive summaries of given papers or paragraphs, focusing on the function of the gene."
LLM_SYSTEM_PROMPT="You are an AI chatbot that helps scientists with their questions. You are very knowledgeable in science, produce truthful answers and are able to explain your reasoning.  "
#DEFAULT_EMBEDDING_LANG_MODEL = os.path.join(LANG_DIR, "bge-large-en-v1_5")
DEVICE_MAP = 'cpu' ## or cuda
N_CTX = 32000 ## Number of tokens of context for the LLM model
CHAT_TYPE = "llama"
MODEL_TYPE = "gguf"