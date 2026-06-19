from fastapi import FastAPI
from pydantic import BaseModel
import tiktoken
import json
import os
from openai import OpenAI
from dotenv import load_dotenv
import time
import csv

app = FastAPI()

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

with open("tiktoken_models.json", "r") as f:
    MODEL_PRICES = json.load(f)

FILE_NAME = "/code/data/sample.json"


class Item(BaseModel):
    text: str
    model: str

class PromptRequest(BaseModel):
    prompt: str
    temperature: float = 0.7
    top_p: float = 0.9

def load_samples():

    if not os.path.exists(FILE_NAME):
        return []

    with open(FILE_NAME, "r") as f:

        try:
            return json.load(f)

        except json.JSONDecodeError:
            return []


def save_samples(samples):

    os.makedirs("/code/data", exist_ok=True)

    with open(FILE_NAME, "w") as f:
        json.dump(samples, f, indent=4)

def log_llm_request(data):

    file_exists = os.path.exists(
        "llm_logs.csv"
    )

    with open(
        "llm_logs.csv",
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        if not file_exists:

            writer.writerow([
                "prompt",
                "temperature",
                "top_p",
                "input_tokens",
                "output_tokens",
                "elapsed_time"
            ])

        writer.writerow([
            data["prompt"],
            data["temperature"],
            data["top_p"],
            data["input_tokens"],
            data["output_tokens"],
            data["elapsed_time"]
        ])

@app.get("/")
def root():
    return {"message": "Token Cost Estimation API"}


@app.get("/models")
def get_models():
    return list(MODEL_PRICES.keys())


# CREATE + TOKEN ESTIMATION
@app.post("/estimate")
def estimate(item: Item):

    if item.model not in MODEL_PRICES:
        return {"error": "Unsupported model"}

    enc = tiktoken.encoding_for_model(item.model)

    token_count = len(enc.encode(item.text))

    cost_per_million = MODEL_PRICES[item.model]["input_cost_per_1m_tokens"]

    cost = (token_count / 1_000_000) * cost_per_million

    samples = load_samples()

    sample = {
        "id": len(samples) + 1,
        "text": item.text,
        "model": item.model,
        "tokens": token_count,
        "cost": cost
    }

    samples.append(sample)

    save_samples(samples)

    return {
        "id": sample["id"],
        "token_count": token_count,
        "token_cost": cost,
        "model": item.model
    }


# READ ALL
@app.get("/samples")
def get_samples():
    return load_samples()


# READ ONE
@app.get("/samples/{sample_id}")
def get_sample(sample_id: int):

    samples = load_samples()

    for sample in samples:

        if sample["id"] == sample_id:
            return sample

    return {"error": "Sample not found"}


# UPDATE
@app.put("/samples/{sample_id}")
def update_sample(sample_id: int, item: Item):

    samples = load_samples()

    for sample in samples:

        if sample["id"] == sample_id:

            enc = tiktoken.encoding_for_model(item.model)

            token_count = len(enc.encode(item.text))

            cost_per_million = MODEL_PRICES[item.model]["input_cost_per_1m_tokens"]

            cost = (token_count / 1_000_000) * cost_per_million

            sample["text"] = item.text
            sample["model"] = item.model
            sample["tokens"] = token_count
            sample["cost"] = cost

            save_samples(samples)

            return sample

    return {"error": "Sample not found"}


# DELETE
@app.delete("/samples/{sample_id}")
def delete_sample(sample_id: int):

    samples = load_samples()

    updated_samples = [
        sample
        for sample in samples
        if sample["id"] != sample_id
    ]

    if len(updated_samples) == len(samples):
        return {"error": "Sample not found"}

    save_samples(updated_samples)

    return {
        "message": "Sample deleted successfully",
        "sample_id": sample_id
    }

@app.post("/generate")
def generate_text(request: PromptRequest):

    start = time.time()

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "user",
                "content": request.prompt
            }
        ],
        temperature=request.temperature,
        top_p=request.top_p
    )

    elapsed_time = round(
        time.time() - start,
        2
    )

    answer = response.choices[0].message.content

    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens

    log_llm_request({
        "prompt": request.prompt,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "elapsed_time": elapsed_time
    })

    return {
        "answer": answer,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "elapsed_time": elapsed_time
    }