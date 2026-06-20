from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict
from generate import generator

app = FastAPI()

# Allow requests from the web UI (and other origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/')
def read_root():
    return {"status": "Server Ready"}


@app.get('/msg/{prompt}')
async def sendStory(prompt: str):
    token = 500
    story = generator(prompt, token)
    return {"Message": story}


@app.post('/generate')
async def generate(payload: Dict):
    """Accepts JSON payload {"prompt": str, "token": int (optional)} and returns generated story."""
    prompt = payload.get('prompt', '')
    token = int(payload.get('token', 150)) if payload.get('token') is not None else 150
    story = generator(prompt, token)
    return {"message": story}

