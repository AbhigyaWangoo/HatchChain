from utils.dataset import DatasetGenerator
from llm.client.gpt import GPTClient
from llm.client.mistral import MistralLLMClient

if __name__ == "__main__":
    # client = GPTClient()
    client = MistralLLMClient()
    generator = DatasetGenerator("rankings.csv", client, 188, "data/fewshotexamples.json")

    generator.generate_dataset()
    print("Done generating dataset!")