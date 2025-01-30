import pathlib

import onnxruntime as ort
import tiktoken
import torch
import torch.nn.functional as F

from sigpt import architecture, config


def generate(
    onnx_path: str | pathlib.Path,
    prompt: str,
    batches: int,
    max_samples: int,
    k: int = 50,
) -> list[str]:
    encoder = tiktoken.get_encoding("gpt2")
    encoded_prompt = torch.tensor(encoder.encode(prompt), dtype=torch.long)
    encoded_prompt = torch.tile(encoded_prompt, (batches, 1))

    ort_session = ort.InferenceSession(onnx_path)
    generated_tokens = generate_tokens(ort_session, encoded_prompt, max_samples, k)

    # The generated response will be invalid if the model had generated and
    # eot_token
    return [
        encoder.decode(truncate_to_eot(i.tolist(), encoder.eot_token)) for i in generated_tokens
    ]


def generate_tokens(
    ort_session: ort.InferenceSession,
    idx: torch.Tensor,
    max_samples: int,
    k: int = 50,
) -> torch.Tensor:
    generations = 0
    with torch.no_grad():
        while generations < max_samples:
            (logits,) = ort_session.run(["output"], {"input": idx.numpy()})  # (B, T, vocab_size)
            # We use the last token for prediction
            logits = torch.tensor(logits[:, -1, :])  # (B, vocab_size)
            probabilities = F.softmax(logits, dim=-1)  # (B, vocab_size)
            top_p, top_p_idx = torch.topk(probabilities, k, dim=-1)  # (B, k)
            gen_idx = torch.multinomial(top_p, num_samples=1)  # (B, 1)
            next_token = torch.take_along_dim(top_p_idx, gen_idx, dim=1)  # (B, 1)

            idx = torch.cat((idx, next_token), dim=-1)
            generations += 1

    return idx


def truncate_to_eot(tokens: list[int], eot_token: int) -> list[int]:
    return tokens[: tokens.index(eot_token)] if eot_token in tokens else tokens


def _save_to_onnx(state_path: str, output_path: str) -> None:
    state = torch.load(state_path, map_location=torch.device("cpu"), weights_only=True)
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}

    model = architecture.Transformer(config.get_gpt_config())
    model.load_state_dict(state)
    model.eval()

    prompt = "Hello, world!"
    encoder = tiktoken.get_encoding("gpt2")
    example_input = torch.tensor(encoder.encode(prompt), dtype=torch.long)[None]

    torch.onnx.export(
        model,
        (example_input,),
        output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {
                0: "batch",
                1: "time",
            },
            "output": {
                0: "batch",
                1: "time",
            },
        },
    )
