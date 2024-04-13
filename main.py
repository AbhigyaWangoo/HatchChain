from utils.dataset import DatasetGenerator
from llm.client.gpt import GPTClient
from llm.client.runpod import RunPodClient
from llm.client.mistral import MistralLLMClient
from llm.client.claude import ClaudeClient
from llm.client.hugging_face import HuggingFaceClient

from utils.graph import MetricGrapher

import multiprocessing

CLIENTS = {
    "gpt4": GPTClient(),
    "mistral": MistralLLMClient(),
    "llama2": RunPodClient(),
    "claude": ClaudeClient(),
}


def gen_dataset(name: str, job_id: int):
    """Worker function to help generate dataset"""
    client_name = name.split("-")[-1]
    client = CLIENTS[client_name]

    generator = DatasetGenerator(
        f"rankings-{name}.json", client, job_id, "data/fewshot_summarized.json"
    )
    generator.generate_dataset()


def multiproc_runall():
    """Runs all dataset generations based upon the provided clients"""
    multiprocessing.set_start_method("spawn")

    procs = []
    # Spawn subprocesses
    for key, _ in CLIENTS.items():
        name = key

        process = multiprocessing.Process(
            target=gen_dataset, args=(f"frontend-{name}",)
        )
        procs.append(process)
        process.start()

    # Wait for all processes to complete
    for process in procs:
        process.join()


if __name__ == "__main__":
    grapher = MetricGrapher("data/evals/clean_evals.csv")
    grapher.generate_relevancy_score_graph()
