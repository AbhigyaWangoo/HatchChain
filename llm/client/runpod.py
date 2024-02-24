from . import base
import time
from enum import StrEnum
import requests
import json
from json import JSONDecodeError
import os
from dotenv import load_dotenv
from urllib.parse import urljoin

RUN_ENDPOINT = "run"
RUNSYNC_ENDPOINT = "runsync"

OUTPUT = "output"
TEXT = "text"
STATUS = "status"


class RunPodStatus(StrEnum):
    IN_QUEUE = "IN_QUEUE"
    COMPLETED = "COMPLETED"
    IN_PROGRESS = "IN_PROGRESS"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMED_OUT = "TIMED_OUT"


MAX_RETRIES = 100


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

        self._key = os.environ.get("RUNPOD_API_KEY")
        self._model_endpoint = os.environ.get("MODEL_ENDPOINT")
        self._port = os.environ.get("MODEL_PORT")

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

    def query(self, prompt: str, full_resp: bool = False) -> str:
        """
        Creates and routes a llama2 job for the runpod
        serverless instance. Returns the response to the runpod op.
        """

        timeout = 100
        url = urljoin(self._model_endpoint, RUNSYNC_ENDPOINT)

        headers = {"Authorization": str(self._key), "Content-Type": "application/json"}

        payload = {
            "input": {
                "prompt": prompt,
                "sampling_params": {
                    "max_tokens": 1000,
                    "n": 1,
                    "presence_penalty": 0.2,
                    "frequency_penalty": 0.7,
                    "temperature": 0.3,
                },
            }
        }

        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if response.status_code != 200:
            print(f"Sending prompt to llama model failed, code: {response.status_code}")
            return

        # Loading the response from the runpod instance
        response_json = json.loads(response.text)

        status = RunPodStatus(response_json[STATUS])
        status_url = urljoin(self._model_endpoint, f"{STATUS}")
        status_url = f"{status_url}/{response_json['id']}"

        for i in range(MAX_RETRIES):
            if status == RunPodStatus.IN_QUEUE or status == RunPodStatus.IN_PROGRESS:
                time.sleep(2)
                try:
                    resp = requests.get(status_url, headers=headers, timeout=1)
                    response_json = json.loads(resp.text)
                    status = response_json[STATUS]
                except JSONDecodeError:
                    print(
                        f"Getting {resp.status_code} on endpoint {status_url}. Retrying..."
                    )

                if i == MAX_RETRIES - 1:
                    status = RunPodStatus.TIMED_OUT

            elif (
                status == RunPodStatus.CANCELLED
                or status == RunPodStatus.FAILED
                or status == RunPodStatus.TIMED_OUT
            ):
                raise ConnectionError(
                    f"Runpod client job had error {status}. Please try a different client at this time"
                )

        if status == RunPodStatus.TIMED_OUT:
            raise ConnectionError(
                "Runpod client job timed out. Please try a different client at this time"
            )
        elif full_resp:
            return response_json

        if OUTPUT in response_json and TEXT in response_json[OUTPUT]:
            if status == RunPodStatus.COMPLETED and isinstance(
                response_json[OUTPUT][TEXT], list
            ):
                return response_json[OUTPUT][TEXT][0]

            return response_json[OUTPUT][TEXT]
        else:
            raise ValueError(
                f"response from runpod was unexpected. response: {response_json}."
            )
