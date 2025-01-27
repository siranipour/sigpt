import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F


def generate(
    model: nn.Module,
    prompt: str,
    batches: int,
    max_samples: int,
    k: int = 50,
    encoder: tiktoken.Encoding | None = None,
) -> list[str]:
    encoder = encoder or tiktoken.get_encoding("gpt2")
    encoded_prompt = torch.tensor(encoder.encode(prompt), dtype=torch.long)
    encoded_prompt = torch.tile(encoded_prompt, (batches, 1))

    generated_tokens = generate_tokens(model, encoded_prompt, max_samples, k)

    # The generated response will be invalid if the model had generated and
    # eot_token
    return [
        encoder.decode(truncate_to_eot(i.tolist(), encoder.eot_token)) for i in generated_tokens
    ]


def generate_tokens(
    model: nn.Module,
    idx: torch.Tensor,
    max_samples: int,
    k: int = 50,
) -> torch.Tensor:
    generations = 0
    with torch.no_grad():
        model.eval()
        while generations < max_samples:
            logits = model(idx)  # (B, T, vocab_size)
            # We use the last token for prediction
            logits = logits[:, -1, :]  # (B, vocab_size)
            probabilities = F.softmax(logits, dim=-1)  # (B, vocab_size)
            top_p, top_p_idx = torch.topk(probabilities, k, dim=-1)  # (B, k)
            gen_idx = torch.multinomial(top_p, num_samples=1)  # (B, 1)
            next_token = torch.take_along_dim(top_p_idx, gen_idx, dim=1)  # (B, 1)

            idx = torch.cat((idx, next_token), dim=-1)
            generations += 1

    return idx


def truncate_to_eot(tokens: list[int], eot_token: int) -> list[int]:
    return tokens[: tokens.index(eot_token)] if eot_token in tokens else tokens
