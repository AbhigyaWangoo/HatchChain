from huggingface_hub import get_inference_endpoint, InferenceEndpoint
from . import base

ZEPHYR_ENDPOINT_NAME = "zephyr-7b-beta-kas"
ZEPHYR_QNA_ENDPOINT_NAME = "zephyr-7b-qna-uht"
MISTRAL_ENDPOINT_NAME="mixtral-8x7b-instruct-v0-1-tgo"


class HuggingFaceClient(base.AbstractLLM):
    """
    A client module to call the Hugging Face API. Currently just have zephyr-7b-beta deployed.
    """

    def __init__(self, inference_endpoint: str = MISTRAL_ENDPOINT_NAME) -> None:
        super().__init__()

        self.inference_client = self.get_endpoint(inference_endpoint)

    def get_endpoint(self, inference_endpoint: str) -> InferenceEndpoint:
        """
        Retrieves inference endpoint, no matter it's state.

        inference_endpoint: The endpoint to retrieve a client for
        """

        endpoint = get_inference_endpoint(inference_endpoint, token="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")

        try:
            endpoint.resume()
        except Exception:
            pass

        return endpoint.wait()

    def query(self, prompt: str, one_off: bool = True) -> str:
        """
        A simple wrapper to the huggingface api

        one_off: boolean to indicate whether the machine should be shut off
        after prompt. For testing here and there.
        """

        output = self.inference_client.client.text_generation(prompt)

        if one_off:
            self.inference_client.pause()

        return output
