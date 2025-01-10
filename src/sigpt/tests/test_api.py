from fastapi.testclient import TestClient

from sigpt import serve

BASE_PROMPT = "Hello, I am a language model and"

def test_api():
    client = TestClient(serve.app)

    response = client.post(f"/sigpt?prompt={BASE_PROMPT}")
    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 1
    assert not (payload[0].keys() ^ set(("input",  "output")))

    n_batches = 1
    response_multiple_batches = client.post(
        f"/sigpt?prompt={BASE_PROMPT}&batches={n_batches}"
    )
    assert len(response_multiple_batches.json()) == n_batches
    response_exceeds_max_batch = client.post(
        f"/sigpt?prompt={BASE_PROMPT}&batches={serve.MAX_BATCHES + 1}"
    )
    assert response_exceeds_max_batch.status_code == 422
