#!/usr/bin/env python3
"""
Simple standalone script to run the chatbot UI.
This can be used to test the chatbot UI without running the full system.
"""

import sys
import argparse
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import os
import random
import time

# Create FastAPI app
app = FastAPI(title="Neuroplastic Qwen Chat", docs_url="/docs", redoc_url="/redoc")

# Mount static files directory
static_dir = os.path.join(os.path.dirname(__file__), "api/static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
else:
    print(f"Error: Static directory not found at {static_dir}")
    sys.exit(1)

# Mock data for simulations
MOCK_STATS = {
    "total_requests": 0,
    "total_tokens": 0,
    "avg_latency": 0.5,
    "tokens_per_second": 35
}

# Models
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    lora_id: str = None
    stop_sequences: list = None

class GenerateResponse(BaseModel):
    response: str
    prompt: str
    model: str
    lora_id: str = None
    stats: dict = {}

# Routes
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    # Redirect to the chatbot UI
    return HTMLResponse(content='<html><head><meta http-equiv="refresh" content="0;url=/static/index.html"></head></html>')

@app.post("/v1/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate text response (mock implementation)."""
    # For standalone testing - generate a mock response
    # In a real implementation, this would call the actual model
    
    responses = [
        f"I understand you're asking about '{request.prompt}'. Let me think about that...",
        f"That's an interesting question about '{request.prompt}'. Here's what I know...",
        f"When it comes to '{request.prompt}', there are several perspectives to consider...",
        f"Thanks for asking about '{request.prompt}'. This is a fascinating topic...",
    ]
    
    # Simulate processing delay
    import asyncio
    await asyncio.sleep(1)  # Simulate model thinking time
    
    # Update mock stats
    latency = random.uniform(0.3, 1.5)
    tokens_generated = random.randint(15, 50)
    tokens_per_second = tokens_generated / latency
    
    # Update global stats
    global MOCK_STATS
    MOCK_STATS["total_requests"] += 1
    MOCK_STATS["total_tokens"] += tokens_generated
    MOCK_STATS["avg_latency"] = (MOCK_STATS["avg_latency"] * (MOCK_STATS["total_requests"] - 1) + latency) / MOCK_STATS["total_requests"]
    MOCK_STATS["tokens_per_second"] = MOCK_STATS["total_tokens"] / (MOCK_STATS["total_requests"] * MOCK_STATS["avg_latency"])
    
    return GenerateResponse(
        response=random.choice(responses) + " This is a mock response for testing the UI.",
        prompt=request.prompt,
        model="Mock-Qwen-Model",
        lora_id=request.lora_id,
        stats={
            "latency": latency,
            "tokens_generated": tokens_generated,
            "tokens_per_second": round(tokens_per_second, 2),
        }
    )

@app.get("/v1/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": True,
        "active_loras": 2,
        "lora_list": ["mock-adapter-1", "mock-adapter-2"],
        "test_generation": {
            "passed": True,
            "latency": 0.1,
            "model": "Mock-Qwen-Model",
            "response": "Hello! This is a test response."
        },
        "stats": MOCK_STATS,
        "model_info": {
            "model_name": "Mock-Qwen-Model",
            "engine_type": "vLLM",
            "max_model_len": 8192,
            "lora_enabled": True
        },
        "gpu_memory_utilization": 0.45
    }

@app.get("/v1/model_info")
async def model_info():
    """Model information endpoint."""
    return {
        "model_name": "Mock-Qwen-Model",
        "model_dtype": "fp16",
        "max_model_len": 8192,
        "engine_type": "vLLM",
        "lora_enabled": True,
        "max_loras": 10,
        "max_lora_rank": 32,
        "active_loras": 2,
        "gpu_memory_utilization": 0.45,
        "stats": MOCK_STATS
    }

@app.get("/v1/lora")
async def list_loras():
    """List available LoRA adapters (mock implementation)."""
    return {
        "adapters": ["mock-adapter-1", "mock-adapter-2"],
        "count": 2
    }

def main():
    parser = argparse.ArgumentParser(description="Run the Neuroplastic Qwen chatbot UI")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    print(f"Starting chatbot UI on http://{args.host}:{args.port}")
    print("This is a standalone version for UI testing only.")
    print("To connect to a real model, use the full Neuroplastic Qwen system.")
    
    uvicorn.run(
        "start_chatbot:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )

if __name__ == "__main__":
    main() 