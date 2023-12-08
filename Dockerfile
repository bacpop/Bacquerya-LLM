FROM nvcr.io/nvidia/cuda:12.3.1-base-ubuntu22.04

RUN mkdir /app

COPY . /app

WORKDIR /app

RUN apt-get update
RUN apt-get install python3.11 python3-pip -y

## Install Python dependencies
RUN pip install poetry
RUN poetry install

RUN pip install -r /app/requirements.txt

## Install llama with cuBLAS
RUN CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
RUN export LLAMA_CUBLAS=1

## set LD_LIBRARY_PATH for SLURM
RUN export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/hps/software/spack/opt/spack/linux-rocky8-cascadelake/gcc-11.2.0/cuda-11.8.0-yahvkfc4w3re2xnjepiu6hslxvvx7cmz/lib64

## Download the language model
RUN mkdir ./language_models
RUN python3 ./run_download_llm.py \
    --repo_id TheBloke/Mistral-7B-SciPhi-32k-GGUF \
    --revision c8e8ef4a096a4d516c88fa22951118b87079d454 \
    --allow_patterns '*Q5_K_M.gguf' \
    --local_dir ./language_models

## SlURM specific set-up
CMD ["module", "load", "cuda"]
CMD ["poetry", "shell"]