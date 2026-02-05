# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="GSAI-ML/LLaDA-1.5", trust_remote_code=True)
messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe(messages)