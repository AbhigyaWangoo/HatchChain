from utils.dataset import DatasetGenerator
from llm.client.gpt import GPTClient
from llm.client.runpod import RunPodClient
from llm.client.mistral import MistralLLMClient
from llm.client.claude import ClaudeClient

from query_engine.src.db import postgres_client

if __name__ == "__main__":
    client = postgres_client.PostgresClient(135)
    client.assign_random_bindings()
    
    # generator = DatasetGenerator("rankings-claude.json", client, 188, "data/fewshotexamples.json")
    # generator.generate_dataset()
    # print("Done generating dataset!")