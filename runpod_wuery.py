import requests
from requests import Request
import json
import time
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from urllib.parse import urljoin
from dotenv import load_dotenv
import os
from llm.runpod import RUN_ENDPOINT, RUNSYNC_ENDPOINT

load_dotenv()
app = FastAPI()

MODEL_ENDPOINT=None
KEY=None

class UnicornException(Exception):
    def __init__(self, name: str):
        self.name = name

@app.exception_handler(UnicornException)
async def unicorn_exception_handler(request: Request, exc: UnicornException):
    print(request.json())

@app.get("/models")
def models_endpoint():
    # TODO figure out what tf to respond to
    print()

# Code taken from here: https://docs.runpod.io/reference/llama2-7b-chat
@app.get("/")
def query_runpod(prompt: str, sync_job: bool=True):
    """
    Creates and routes a llama2 job for the runpod 
    serverless instance. Returns the response to the runpod op.
    """

    if sync_job:
        url = urljoin(MODEL_ENDPOINT, RUNSYNC_ENDPOINT)
        timeout=100
    else:
        url = urljoin(MODEL_ENDPOINT, RUN_ENDPOINT)
        timeout=5

    headers = {
        "Authorization": str(KEY),
        "Content-Type": "application/json"
    }

    payload = {
        "input": {
            "prompt": prompt,
            "sampling_params": {
                "max_tokens": 1000,
                "n": 1,
                "presence_penalty": 0.2,
                "frequency_penalty": 0.7,
                "temperature": 0.3,
            }
        }
    }

    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if response.status_code != 200:
        print(f"Sending prompt to llama model failed, code: {response.status_code}")
        return

    response_json = json.loads(response.text)
    if sync_job:
        return response_json

    # TODO async is not properly reading via stream. Need to modify
    # to keep reading until stream is finished
    status_url = urljoin(MODEL_ENDPOINT, f"stream/{response_json['id']}")
    for _ in range(10):
        time.sleep(1)
        get_status = requests.get(status_url, headers=headers, timeout=5)
        print(get_status.text)

if __name__ == "__main__":
    KEY=os.environ.get('RUNPOD_API_KEY')
    MODEL_ENDPOINT=os.environ.get('MODEL_ENDPOINT')
    PORT=os.environ.get('MODEL_PORT')

    if KEY is None:
        print("Please set RUNPOD_API_KEY in .env file")
    elif MODEL_ENDPOINT is None:
        print("Please set MODEL_ENDPOINT in .env file")
    elif PORT is None:
        print("Please set MODEL_PORT in .env file")
    else:
        uvicorn.run(app, port=int(PORT), host="0.0.0.0")
