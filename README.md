# ExplainableClassifier
An LLM-based classifier capable of providing explanations for resume classification.

### Setup
Setup a .env file with the following env vars set
```
RUNPOD_API_KEY
MODEL_ENDPOINT
MODEL_PORT
EXPLAINABLE_CLASSIFIER_ENDPOINT
MISTRAL_API_KEY
HUGGING_FACE_TOKEN
CLAUDE_KEY
OPENAI_API_KEY
```

Then, install all required dependencies.
```
pip3 install -r requirements.txt
git submodule update --init
```

And run the server with ```python3 server.py``` to get started. HatchChain expects >=python3.11
