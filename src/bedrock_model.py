
import boto3

import os
from dotenv import load_dotenv

load_dotenv()


def ask_bedrock(prompt: str) -> str:
    """Send a prompt to Amazon Nova Lite and return its answer."""

    client = boto3.client(
        service_name="bedrock-runtime",
        region_name=os.getenv("AWS_REGION", "us-east-1")
    )

    response = client.converse(
        modelId=os.getenv(
        "BEDROCK_MODEL_ID",
        "amazon.nova-lite-v1:0"
    ),
        messages=[
            {
                "role": "user",
                "content": [{"text": prompt}]
            }
        ],
        inferenceConfig={
            "maxTokens": 300,
            "temperature": 0.3
        }
    )

    return response["output"]["message"]["content"][0]["text"]


def ask_model(prompt: str, provider: str = None) -> str:
    """Choose between the local Ollama model and Amazon Bedrock."""

    if provider is None:
        provider = os.getenv("AI_PROVIDER", "ollama").lower()

    if provider == "bedrock":
        return ask_bedrock(prompt)

    elif provider == "ollama":
        import ollama

        response = ollama.chat(
            model="llama3.2",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return response["message"]["content"]

    else:
        raise ValueError(
            "AI_PROVIDER must be 'ollama' or 'bedrock'."
        )
