"""Minimal FastAPI gateway wrapping the serving engine.

Only essential endpoints are provided to ensure the project runs
end-to-end without the heavier full-featured gateway.
"""
from __future__ import annotations

import asyncio
import time
import os
from typing import Dict, Any, List

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import ValidationError
import structlog

from api.models import GenerateRequest, GenerateResponse
from serving.base_engine import BaseServingEngine

logger = structlog.get_logger(__name__)


class NeuroplasticAPI:
    """Lightweight API gateway exposing generate and health routes."""

    def __init__(self, config, serving_engine: BaseServingEngine):
        self.config = config
        self.serving_engine = serving_engine
        self.app = FastAPI(
            title="Neuroplastic-Qwen API",
            version="0.1.0",
            docs_url="/docs",
            redoc_url="/redoc",
        )
        
        # Mount static files directory for the chatbot UI
        static_dir = os.path.join(os.path.dirname(__file__), "static")
        if os.path.exists(static_dir):
            self.app.mount("/static", StaticFiles(directory=static_dir), name="static")
        else:
            # Create the directory if it doesn't exist
            os.makedirs(static_dir, exist_ok=True)
            self.app.mount("/static", StaticFiles(directory=static_dir), name="static")
            
        self._routes()

    # ------------------------------------------------------------------
    def _routes(self):
        @self.app.get("/", response_class=HTMLResponse)
        async def root(request: Request):
            # Redirect to the chatbot UI
            return HTMLResponse(content='<html><head><meta http-equiv="refresh" content="0;url=/static/index.html"></head></html>')

        @self.app.post("/v1/generate", response_model=GenerateResponse)
        async def generate(request: GenerateRequest, background_tasks: BackgroundTasks):
            start_ts = time.time()
            try:
                response_text = await self.serving_engine.generate(
                    prompt=request.prompt,
                    lora_id=request.lora_id,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    stop_sequences=request.stop_sequences,
                )
            except (ValueError, ValidationError) as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logger.error("Generation failed", error=str(e))
                raise HTTPException(status_code=500, detail="Generation failed")

            latency = time.time() - start_ts
            resp = GenerateResponse(
                response=response_text,
                prompt=request.prompt,
                model=self.config.model.name,
                lora_id=request.lora_id,
                stats={
                    "latency": round(latency, 3),
                    "tokens_generated": len(response_text.split()),
                    "tokens_per_second": round(len(response_text.split()) / max(latency, 1e-6), 2),
                },
            )
            # Background logging or Kafka trace could be added here.
            return resp

        @self.app.get("/v1/health")
        async def health() -> Dict[str, Any]:
            return await self.serving_engine.health_check()
            
        @self.app.get("/v1/lora")
        async def list_loras() -> Dict[str, Any]:
            """List available LoRA adapters."""
            try:
                adapters = await self.serving_engine.list_loras()
                return {
                    "adapters": adapters,
                    "count": len(adapters)
                }
            except Exception as e:
                logger.error("Failed to list LoRA adapters", error=str(e))
                raise HTTPException(status_code=500, detail="Failed to list LoRA adapters")


