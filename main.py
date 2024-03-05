from utils.dataset import DatasetGenerator
from llm.client.gpt import GPTClient

if __name__ == "__main__":
    client = GPTClient()
    generator = DatasetGenerator("rankings.csv", client, 188)

    generator.generate_dataset()
    print("Done generating dataset!")