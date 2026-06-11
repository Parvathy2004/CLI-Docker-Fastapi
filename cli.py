from fastapi import FastAPI
from pydantic import BaseModel
import tiktoken
import json
import os

app = FastAPI()

# Load model prices
with open("tiktoken_models.json", "r") as f:
    MODEL_PRICES = json.load(f)


class Item(BaseModel):
    text: str
    model: str


class SampleItem(BaseModel):
    id: int
    text: str
    model: str
    tokens: int
    cost: float


FILE_NAME = "sample.json"


def load_samples():
    if not os.path.exists(FILE_NAME):
        return []

    with open(FILE_NAME, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def save_samples(samples):
    with open(FILE_NAME, "w") as f:
        json.dump(samples, f, indent=4)


@app.get("/")
def root():
    return {"message": "Token Cost Estimation API"}


@app.get("/models")
def get_models():
    return list(MODEL_PRICES.keys())


# CREATE + TOKEN ESTIMATION
@app.post("/estimate")
def token_cost(item: Item):

    if item.model not in MODEL_PRICES:
        return {"error": "Unsupported model"}

    enc = tiktoken.encoding_for_model(item.model)

    token_count = len(enc.encode(item.text))

    cost_per_million = MODEL_PRICES[item.model]["input_cost_per_1m_tokens"]

    cost = (token_count / 1_000_000) * cost_per_million

    samples = load_samples()

    sample_id = len(samples) + 1

    sample = {
        "id": sample_id,
        "text": item.text,
        "model": item.model,
        "tokens": token_count,
        "cost": cost
    }

    samples.append(sample)

    save_samples(samples)

    return {
        "id": sample_id,
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