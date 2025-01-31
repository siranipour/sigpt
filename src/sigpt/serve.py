import dataclasses

from better_profanity import profanity
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from sigpt import config, inference

MAX_BATCHES = 10
MAX_GEN_LEN = 100


@dataclasses.dataclass
class ModelResponse:
    input: str
    output: str


app = FastAPI()

origins = [
    "https://siranipour.io",
]
origins_regex = "http://localhost:.*"


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=origins_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/sigpt/")
async def generate_tokens(
    prompt: str,
    batches: int = 1,
    max_len: int = 20,
) -> list[ModelResponse]:
    check_request(prompt, batches, max_len)
    generated = inference.generate(config.get_onnx_path(), prompt, batches, max_len)
    return [ModelResponse(prompt, gen.lstrip(prompt)) for gen in generated]


def check_request(prompt: str, batches: int, max_len: int) -> None:
    if max_len > MAX_GEN_LEN:
        raise HTTPException(
            status_code=422,
            detail=f"Max requested sequence generation length cannot exceed {MAX_GEN_LEN}",
        )
    if batches > MAX_BATCHES:
        raise HTTPException(
            status_code=422,
            detail=f"Max requested batches cannot exceed {MAX_BATCHES}",
        )
    if profanity.contains_profanity(prompt):
        raise HTTPException(
            status_code=422,
            detail="Input prompt is not valid",
        )
