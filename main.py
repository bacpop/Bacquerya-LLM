## This project is about implementing a LLM model for searching PubMed papers
## Input: texts (paragraphs) from PubMed papers ; query (gene, species)
## Output: summary of the texts (paragraphs) with relevance scores to the query


def start_language_models():
    ### Load LLM models
    from config import DEFAULT_SUMMARISER_LANG_MODEL, DEFAULT_EMBEDDING_LANG_MODEL, DEVICE_MAP
    from lib.summariser import Summariser
    lang_model = DEFAULT_SUMMARISER_LANG_MODEL
    emb_model = DEFAULT_EMBEDDING_LANG_MODEL
    device = DEVICE_MAP
    summariser_obj = Summariser(
        summariser_model_path=lang_model,
        embedding_model_path=emb_model,
        device_map=device
    )
    summariser_obj.load_models()
    return summariser_obj


summariser = start_language_models()

### Start the API
from fastapi import FastAPI
from typing import Optional, List, AnyStr
from pydantic import BaseModel

app = FastAPI()


class Text2Summarise(BaseModel):
    query: AnyStr
    texts: List[AnyStr]
    #text: Optional[AnyStr]
    temperature: Optional[float]
    max_length: Optional[int]
    min_length: Optional[int]
    do_sample: Optional[bool]


@app.get("/")
def root():
    return {"message": "welcome to Local LLM summariser"}


@app.post("/summarise")
def call_summariser(text2summarise: Text2Summarise):
    #summariser.get_query_embedding(text2summarise.query)
    output_lst = []
    for text in text2summarise.texts:
        text_summary = summariser.summarise(
         text=text,
         max_length=text2summarise.max_length,
         min_length=text2summarise.min_length,
         do_sample=text2summarise.do_sample,
         temperature=text2summarise.temperature
        )
        output_lst.append(
            {"text": text,
             "summary": text_summary,
             #"score": summariser.get_similarity_score(text_summary)
             })
    return output_lst

