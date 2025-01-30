# SIGPT 🤖

My attempt at training a transformer based LLM model. The environment for interacting
with this code base is managed by `uv`.

## Inference

The model can be interacted with using the package's `fastAPI` server. The
`onnx` artefact must be fetched from the GCP bucket in order to run the server
locally. To start the server simply run:


```prompt
uv run uvicorn sigpt.serve:app --port 8000 --host 0.0.0.0
```

which will accept `POST` requests

```prompt
curl -X POST "http://0.0.0.0:8000/sigpt/?prompt=Hello%20world\!&batches=3"
```


## Training

The model was trained on 8x A100 GPUs using
[LambdaLabs](https://lambdalabs.com) using the
[allenai/c4-en](https://huggingface.co/datasets/allenai/c4) dataset. The
training script can be invoked (assuming distributed workloads) using:

```prompt
uv run --group train torchrun --standalone --nproc_per_node=8 scripts/train.py
```
