from utils.dataset import DatasetGenerator
from llm.client.gpt import GPTClient
from llm.client.runpod import RunPodClient
from llm.client.mistral import MistralLLMClient
from llm.client.claude import ClaudeClient

if __name__ == "__main__":
    # client = GPTClient() # GPT4
    # client = MistralLLMClient() # MISTRAL
    client=ClaudeClient()# Claude 3
    # client = RunPodClient() # Llama 2
    # print(cclient.query("Hello! How are you today?"))
    generator = DatasetGenerator("rankings-claude.json", client, 188, "data/fewshotexamples.json")

    generator.generate_dataset()
    print("Done generating dataset!")