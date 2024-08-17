import dataclasses
from typing import Optional

from fastapi import FastAPI

from sigpt.model import architecture, sample

@dataclasses.dataclass
class ModelResponse:
    input: str
    output: str

app = FastAPI()

@app.get("/sigpt/")
def generate_tokens(
    prompt: str, batches: Optional[int]=1, max_len: Optional[int]=20,
) -> list[ModelResponse]:
    mdl = architecture.Transformer(architecture.GPTConfig)
    generated = sample.generate(mdl, prompt, batches, max_len)
    return [ModelResponse(prompt, gen.lstrip(prompt)) for gen in generated]