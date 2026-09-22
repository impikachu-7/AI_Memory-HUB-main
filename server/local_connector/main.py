"""AI Memory Hub Local Connector for a user's local Ollama instance.

Run with: uvicorn local_connector.main:app --host 127.0.0.1 --port 8765
"""
import json
import os
from collections.abc import AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
ALLOWED_ORIGINS = {origin.strip() for origin in os.getenv("CONNECTOR_ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",") if origin.strip()}
TIMEOUT = httpx.Timeout(float(os.getenv("CONNECTOR_TIMEOUT_SECONDS", "120")))
app = FastAPI(title="AI Memory Hub Local Connector", version="1.0.0")


class ChatRequest(BaseModel):
    model: str = Field(min_length=1)
    messages: list[dict]
    stream: bool = True
    options: dict | None = None


app.add_middleware(
    CORSMiddleware,
    allow_origins=list(ALLOWED_ORIGINS),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Access-Control-Request-Private-Network"],
)

@app.middleware("http")
async def add_local_network_permission_header(request, call_next):
    response = await call_next(request)
    if request.headers.get("access-control-request-private-network") == "true":
        response.headers["Access-Control-Allow-Private-Network"] = "true"
    return response


async def ollama_get(path: str):
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(f"{OLLAMA_URL}{path}")
            response.raise_for_status()
            return response.json()
    except httpx.ConnectError as exc:
        raise HTTPException(503, "Ollama is not running") from exc
    except httpx.TimeoutException as exc:
        raise HTTPException(504, "Ollama request timed out") from exc
    except httpx.HTTPStatusError as exc:
        raise HTTPException(502, "Ollama returned an error") from exc


@app.get("/health")
async def health():
    try:
        await ollama_get("/api/tags")
        return {"status": "connected", "ollama_url": OLLAMA_URL}
    except HTTPException as exc:
        return {"status": "disconnected", "code": "OLLAMA_OFFLINE", "detail": str(exc.detail)}


@app.get("/status")
async def status():
    health_result = await health()
    return {"connected": health_result["status"] == "connected", **health_result}


@app.get("/models")
async def models():
    data = await ollama_get("/api/tags")
    return {"models": [{"name": model["name"], "size": model.get("size"), "modified_at": model.get("modified_at")} for model in data.get("models", [])]}


async def stream_chat(body: ChatRequest) -> AsyncIterator[str]:
    payload = body.model_dump(exclude_none=True)
    payload["stream"] = True
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            async with client.stream("POST", f"{OLLAMA_URL}/api/chat", json=payload) as response:
                if response.status_code == 404:
                    raise HTTPException(404, "Ollama model is not installed")
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        yield line + "\n"
    except HTTPException:
        raise
    except httpx.ConnectError as exc:
        raise HTTPException(503, "Ollama is not running") from exc
    except httpx.TimeoutException as exc:
        raise HTTPException(504, "Ollama request timed out") from exc
    except httpx.HTTPStatusError as exc:
        raise HTTPException(502, "Ollama returned an error") from exc


@app.post("/chat")
async def chat(body: ChatRequest):
    if body.stream:
        return StreamingResponse(stream_chat(body), media_type="application/x-ndjson")
    return await generate(body)


@app.post("/generate")
async def generate(body: ChatRequest):
    payload = body.model_dump(exclude_none=True)
    payload["stream"] = False
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
            if response.status_code == 404:
                raise HTTPException(404, "Ollama model is not installed")
            response.raise_for_status()
            return response.json()
    except HTTPException:
        raise
    except httpx.ConnectError as exc:
        raise HTTPException(503, "Ollama is not running") from exc
    except httpx.TimeoutException as exc:
        raise HTTPException(504, "Ollama request timed out") from exc
    except httpx.HTTPStatusError as exc:
        raise HTTPException(502, "Ollama returned an error") from exc
