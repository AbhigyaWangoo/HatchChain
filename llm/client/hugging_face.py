import requests
from . import base
import os
from dotenv import load_dotenv
import re

load_dotenv()

ZEPHYR_ENDPOINT_NAME = "zephyr-7b-beta-kas"
ZEPHYR_QNA_ENDPOINT_NAME = "zephyr-7b-qna-uht"
MISTRAL_ENDPOINT_NAME = "mixtral-8x7b-instruct-v0-1-tgo"
LLAMA2_ENDPOINT_NAME = (
    "https://sffoi5rv3j9b8ga7.us-east-1.aws.endpoints.huggingface.cloud"
)

HUGGING_FACE_TOKEN = os.environ.get("HUGGING_FACE_TOKEN")

HEADERS = {
    "Accept": "application/json",
    "Authorization": f"Bearer {HUGGING_FACE_TOKEN}",
    "Content-Type": "application/json",
}


def query(payload):
    """
    A simple helper function to send a hugging face response
    """
    response = requests.post(
        LLAMA2_ENDPOINT_NAME, headers=HEADERS, json=payload, timeout=100
    )
    return response.json()


class HuggingFaceClient(base.AbstractLLM):
    """
    A client module to call the Hugging Face API. Currently just have zephyr-7b-beta deployed.
    """

    def __init__(self) -> None:
        super().__init__()

    def recursive_summarizer(
        self, prompt: str, target_length: int, n_chunks: int = 10
    ) -> str:
        """
        Trim the provided prompt in summary chunks until it
        reaches the specified length or less.
        """
        prompt = re.sub(" +", " ", prompt)
        final_res = ""

        # target_length = n_chunks * chunk_size
        chunk_size = target_length // n_chunks

        for chunk in range(n_chunks):
            # Problem is there are way too many tokens in the input. chop them up into smaller pieces.
            trim_section = prompt[chunk * chunk_size : (chunk + 1) * chunk_size]
            output = query(
                {
                    "inputs": f"<s>[INST] <<SYS>> You are a summarizer. Properly abide by the summary limit. </s><s>[INST] Summarize the following information into EXACTLY {chunk_size} chracters or less: {trim_section} [/INST]",
                    "parameters": {"max_new_tokens": chunk_size},
                }
            )
            final_res += output[0]["generated_text"]

        return final_res

    def query(self, prompt: str, context_length: int = 4000) -> str:
        """
        A simple wrapper to the huggingface api

        one_off: boolean to indicate whether the machine should be shut off
        after prompt. For testing here and there.
        """

        for _ in range(10):
            try:
                base_prompt = f"<s>[INST] <<SYS>> You are a helpful recruitment assistant. Properly exaplain your reasonings. </s><s>[INST] {prompt} [/INST]"

                if len(base_prompt) > context_length:
                    base_prompt = self.recursive_summarizer(
                        base_prompt, context_length, 10
                    )

                print(len(base_prompt))

                output = query(
                    {  # Problem is there are way too many tokens in the input. chop them up into smaller pieces.
                        "inputs": base_prompt,
                        "parameters": {"max_new_tokens": 400},
                    }
                )
                if "error" not in output:
                    return output[0]["generated_text"]
                else:
                    print(output)
            except Exception as e:
                print("Exception in hugging face querier")
                print(e)

        return output[0]["generated_text"]
