FROM afgreen/llama-rt:latest

RUN mkdir /app

COPY . /app

WORKDIR /app
RUN pip install -r /app/requirements.txt
