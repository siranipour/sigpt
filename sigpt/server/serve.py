from sigpt.model import sample

from fastapi import FastAPI

app = FastAPI()

@app.get("/sigpt/")
def generate_tokens():
    sample.hello
    return {"Hello": "There"}