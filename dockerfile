FROM python:3.14-slim

WORKDIR /code

COPY requirements.txt /code/requirements.txt

RUN pip install -r /code/requirements.txt

COPY cli.py /code/cli.py
COPY sample.json /code/sample.json
COPY tiktoken_models.json /code/tiktoken_models.json

CMD ["uvicorn", "cli:app", "--host", "0.0.0.0", "--port", "8000"]