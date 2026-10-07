from fastapi import FastAPI
from pydantic import BaseModel
from src.main import run_agent
app = FastAPI()


class ChatRequest(BaseModel):
    model: str
    messages: list


@app.get("/v1/models")
def get_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "my-ai-agent",
                "object": "model",
                "owned_by": "local"
            }
        ]
    }


@app.post("/v1/chat/completions")
def chat_completions(request: ChatRequest):

    # Get the latest user message
    user_message = request.messages[-1]["content"]

    # Send it to your existing AI Agent
    response = run_agent(user_message)

    return {
        "id": "agent-response",
        "object": "chat.completion",
        "model": "my-ai-agent",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response
                },
                "finish_reason": "stop"
            }
        ]
    }