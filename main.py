from utils.dataset import DatasetGenerator
from llm.client.gpt import GPTClient
from llm.client.runpod import RunPodClient
from llm.client.mistral import MistralLLMClient
from llm.client.claude import ClaudeClient

import multiprocessing

CLIENTS = {
    "gpt4": GPTClient(),
    "mistral": MistralLLMClient(),
    "llama2": RunPodClient(),
    "claude": ClaudeClient()
}

def gen_dataset(name: str):
    """ Worker function to help generate dataset """
    client_name = name.split("-")[-1]
    client = CLIENTS[client_name]
    print(client)

    generator = DatasetGenerator(f"rankings-{name}.json", client, 135, "data/fewshotexamples.json")
    generator.generate_dataset()

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')

    procs=[]
    # Spawn subprocesses
    for key, _ in CLIENTS.items():
        name = key

        process = multiprocessing.Process(target=gen_dataset, args=(f"frontend-{name}", ))
        procs.append(process)
        process.start()

    # Wait for all processes to complete
    for process in procs:
        process.join()
