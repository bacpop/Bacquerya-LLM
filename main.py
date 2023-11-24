## This project is about implementing a LLM model for searching PubMed papers
## Input: texts (paragraphs) from PubMed papers ; query (gene, species)
## Output: summary of the texts (paragraphs) with relevance scores to the query


def start_language_models(lang_model_path=None, emb_model_path=None, device=None, chat_type=None, n_ctx=None, task=''):
    ### Load LLM models
    from config import DEFAULT_SUMMARISER_LANG_MODEL, DEFAULT_EMBEDDING_LANG_MODEL, DEVICE_MAP, CHAT_TYPE, N_CTX
    from lib.summariser import LLM
    lang_model_path = DEFAULT_SUMMARISER_LANG_MODEL if lang_model_path is None else lang_model_path
    emb_model_path = DEFAULT_EMBEDDING_LANG_MODEL if emb_model_path is None else emb_model_path
    device = DEVICE_MAP if device is None else device
    chat_type = CHAT_TYPE if chat_type is None else chat_type
    n_ctx = N_CTX if n_ctx is None else n_ctx

    language_model = LLM(
        main_model_path=lang_model_path,
        embedding_model_path=emb_model_path,
        device_map=device,
    )
    language_model.load_main_LLM(n_ctx=n_ctx, model_type=chat_type, task=task)
    language_model.load_embedding_model()

    return language_model


summariser_model = start_language_models(task="summarization")

### Start the API
from fastapi import FastAPI
from typing import Optional, List, AnyStr
from pydantic import BaseModel
import asyncio

app = FastAPI()


class Text2Summarise(BaseModel):
    query: AnyStr
    texts: List[AnyStr]
    temperature: Optional[float] = 0.2
    max_length: Optional[int] = 100
    min_length: Optional[int] = 20
    do_sample: Optional[bool] = False
    re_summarise: Optional[bool] = True
    similarity_threshold: Optional[float] = 0.3
    combine_summaries: Optional[bool] = True


@app.get("/")
async def root():
    return {"message": "welcome to Local LLM summariser"}


@app.post("/summarise")
async def call_summariser(text2summarise: Text2Summarise):
    from lib.summariser import Summariser
    summariser = Summariser(summariser_model, **text2summarise.__dict__)
    return summariser.process()
