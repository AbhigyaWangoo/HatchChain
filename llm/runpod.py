from . import base
import time
import requests
import json
import os
from dotenv import load_dotenv
from urllib.parse import urljoin

RUN_ENDPOINT="run"
RUNSYNC_ENDPOINT="runsync"
 
class RunPodClient(base.AbstractLLM):
    def __init__(self) -> None:
        super().__init__()
        self._key = None
        self._model_endpoint = None
        self._port = None

        if not self.__set_env():
            exit(1)

    def __set_env(self) -> bool:
        load_dotenv()

        self._key = os.environ.get('RUNPOD_API_KEY')
        self._model_endpoint = os.environ.get('MODEL_ENDPOINT')
        self._port = os.environ.get('MODEL_PORT')

        if self._key is None:
            print("Please set RUNPOD_API_KEY in .env file")
            return False
        elif self._model_endpoint is None:
            print("Please set MODEL_ENDPOINT in .env file")
            return False
        elif self._port is None:
            print("Please set MODEL_PORT in .env file")
            return False

        return True

    def query(self, prompt: str, is_async: bool = False) -> str:
        """
        Creates and routes a llama2 job for the runpod 
        serverless instance. Returns the response to the runpod op.
        """

        if is_async:
            url = urljoin(self._model_endpoint, RUNSYNC_ENDPOINT)
            timeout = 100
        else:
            url = urljoin(self._model_endpoint, RUN_ENDPOINT)
            timeout = 5

        headers = {
            "Authorization": str(self._key),
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

        response = requests.post(url, headers=headers,
                                 json=payload, timeout=timeout)
        if response.status_code != 200:
            print(
                f"Sending prompt to llama model failed, code: {response.status_code}")
            return

        response_json = json.loads(response.text)
        if is_async:
            return response_json

        # TODO async is not properly reading via stream. Need to modify
        # to keep reading until stream is finished
        status_url = urljoin(self._model_endpoint, f"stream/{response_json['id']}")
        for _ in range(10):
            time.sleep(1)
            get_status = requests.get(status_url, headers=headers, timeout=5)
            print(get_status.text)
