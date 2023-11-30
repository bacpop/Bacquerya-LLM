FROM python:3.11.0

RUN mkdir /app

COPY . /app

WORKDIR /app

RUN pip install -r /app/requirements.txt

## Download the language model
RUN mkdir ./language_models

RUN python3 ./run_download_llm.py \
    --repo_id TheBloke/Mistral-7B-SciPhi-32k-GGUF \
    --revision c8e8ef4a096a4d516c88fa22951118b87079d454 \
    --allow_patterns '*Q5_K_M.gguf' \
    --local_dir ./language_models