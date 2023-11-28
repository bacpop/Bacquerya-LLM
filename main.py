## This project is about implementing a LLM model for searching PubMed papers
## Input: texts (paragraphs) from PubMed papers ; query (gene, species)
## Output: summary of the texts (paragraphs) with relevance scores to the query


from lib.llm import start_language_models
summariser_model = start_language_models()

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
