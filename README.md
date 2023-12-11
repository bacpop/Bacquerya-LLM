This project is a part of [BacQuerya](https://www.bacquerya.com) website. This is the Large Language Model backend, that summarises given articles. Feel free to use it, however the repo is at the experimental stage and may raise errors or give unexpected summaries. 

## Installation and Running
### METHOD 1 - in a local venv
1. Clone the repository

`git clone git@github.com:bacpop/Bacquerya-LLM.git`
2. Install the requirements

`pip install -r requirements.txt`

3. Download LLM from [HuggingFace](https://huggingface.co)

`python run_download_llm.py --repo_id TheBloke/SciPhi-Mistral-7B-32k-GGUF --local_dir language_models --allow_patterns "*Q5_K_M*gguf"`

4. Running LLM

a) on server

`uvicorn main:app`

b) or to run locally

`python run_llm_locally.py  --articles_json path/to/articles.json --language_model ./language_models/sciphi-mistral-7b-32k.Q5_K_M.gguf --n_ctx 32000 --device_map cuda --chat_type llama --model_type gguf --output_file summarised_article.txt`

### Method 2 - Docker
Having Docker Engine running

`docker run -v output_dir -v input_dir sbgonenc/summariser-mistal7b-sciphi-32k bash`

and run the model inside the container, with the commands used above. For CUDA acceleration, you can try

`docker run -v output_dir -v input_dir sbgonenc/summariser_mistral-7b-sciphi-32k-gguf_cuda-32k bash`

## Running on SLURM

Currently, I could not figure out how to run the Docker image on SLURM using GPU accelerator. However, I learnt a trick to run how to run it in a poetry shell. Having logged into SLURM and establishing venv:

1. Install poetry

`pip install poetry`

`pip upgrade poetry`

2. In the base folder of the project (same place where pyproject.toml is):

`poetry install`

3. Install other requirements

`pip install -r requirments`

4. Enter the poetry shell

`module load cuda`

5. Install llama.cpp model for gguf models

`CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall --no-cache-dir`

6. Add python path variable for CUDA communication

```export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/hps/software/spack/opt/spack/linux-rocky8-cascadelake/gcc-11.2.0/cuda-11.8.0-yahvkfc4w3re2xnjepiu6hslxvvx7cmz/lib64```

7. Within the venv

`python run_llm_locally.py  --articles_json path/to/articles.json --language_model ./language_models/sciphi-mistral-7b-32k.Q5_K_M.gguf --n_ctx 32000 --device_map cuda --chat_type llama --model_type gguf --output_file summarised_article.txt`
