"""vLLM serving engine implementation for neuroplastic Qwen system."""

import asyncio
import time
import torch
from typing import Dict, Any, Optional, List
from pathlib import Path

try:
    from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
    from vllm.lora.request import LoRARequest
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    AsyncLLMEngine = None
    AsyncEngineArgs = None
    SamplingParams = None
    LoRARequest = None

import structlog
from .base_engine import BaseServingEngine

logger = structlog.get_logger(__name__)


class NeuroplasticVLLMEngine(BaseServingEngine):
    """vLLM-based serving engine with hot-reloadable LoRA support."""

    def __init__(self, config):
        super().__init__(config)
        
        if not VLLM_AVAILABLE:
            raise ImportError(
                "vLLM is not available. Please install with: pip install vllm>=0.2.7"
            )
        
        self.engine: Optional[AsyncLLMEngine] = None
        self.lora_counter = 0
        self.generation_stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "avg_latency": 0.0,
            "last_request_time": None
        }

    async def initialize(self) -> bool:
        """Initialize the vLLM engine."""
        try:
            self.logger.info("Initializing vLLM engine", model=self.config.model.name)
            
            # Create engine arguments
            engine_args = AsyncEngineArgs(
                model=self.config.model.name,
                tokenizer=self.config.model.name,
                dtype=self.config.model.dtype,
                max_model_len=self.config.model.max_model_len,
                gpu_memory_utilization=self.config.serving.gpu_memory_utilization,
                max_num_batched_tokens=self.config.serving.max_num_batched_tokens,
                max_num_seqs=self.config.serving.max_num_seqs,
                enable_lora=self.config.serving.enable_lora,
                max_lora_rank=self.config.serving.max_lora_rank,
                max_loras=self.config.serving.max_loras,
                trust_remote_code=self.config.model.trust_remote_code,
                disable_log_stats=False,
                enforce_eager=False,  # Enable CUDA graphs for better performance
            )
            
            # Create the async engine
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            self.model_loaded = True
            
            self.logger.info("vLLM engine initialized successfully")
            
            # Perform warmup
            await self.warmup()
            
            return True
            
        except Exception as e:
            self.logger.error("Failed to initialize vLLM engine", error=str(e))
            return False

    async def generate(
        self,
        prompt: str,
        lora_id: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """Generate text response using vLLM."""
        if not self.model_loaded or self.engine is None:
            raise RuntimeError("Engine not initialized. Call initialize() first.")
        
        # Validate parameters
        self._validate_generate_params(prompt, max_tokens, temperature, top_p)
        
        start_time = time.time()
        
        try:
            # Create sampling parameters
            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop_sequences,
                **kwargs
            )
            
            # Create LoRA request if specified
            lora_request = None
            if lora_id and lora_id in self.active_loras:
                lora_request = LoRARequest(
                    lora_name=lora_id,
                    lora_int_id=hash(lora_id) % 1000000,  # Simple hash for ID
                    lora_path=self.active_loras[lora_id]
                )
            
            # Generate response
            request_id = f"req_{int(time.time() * 1000000)}"
            
            self.logger.debug(
                "Starting generation",
                request_id=request_id,
                lora_id=lora_id,
                max_tokens=max_tokens
            )
            
            # Submit request to engine
            results_generator = self.engine.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request_id,
                lora_request=lora_request
            )
            
            # Wait for completion
            final_output = None
            async for request_output in results_generator:
                if request_output.finished:
                    final_output = request_output
                    break
            
            if final_output is None or not final_output.outputs:
                raise RuntimeError("No output generated")
            
            response_text = final_output.outputs[0].text
            
            # Update statistics
            latency = time.time() - start_time
            self._update_stats(latency, len(response_text.split()))
            
            self.logger.debug(
                "Generation completed",
                request_id=request_id,
                latency=f"{latency:.3f}s",
                response_length=len(response_text)
            )
            
            return response_text
            
        except Exception as e:
            self.logger.error(
                "Generation failed",
                error=str(e),
                prompt_length=len(prompt),
                lora_id=lora_id
            )
            raise

    async def reload_lora(self, lora_path: str, lora_id: str) -> bool:
        """Hot-reload a LoRA adapter."""
        if not self.model_loaded or self.engine is None:
            self.logger.error("Cannot reload LoRA: engine not initialized")
            return False
        
        try:
            self.logger.info("Reloading LoRA adapter", lora_id=lora_id, path=lora_path)
            
            # Validate LoRA path exists
            if not Path(lora_path).exists():
                self.logger.error("LoRA path does not exist", path=lora_path)
                return False
            
            # Remove existing LoRA if present
            if lora_id in self.active_loras:
                await self.remove_lora(lora_id)
            
            # Add new LoRA
            lora_request = LoRARequest(
                lora_name=lora_id,
                lora_int_id=hash(lora_id) % 1000000,
                lora_path=lora_path
            )
            
            # Note: vLLM doesn't have a direct add_lora method in AsyncLLMEngine
            # The LoRA is loaded on-demand during generation
            self.active_loras[lora_id] = lora_path
            
            self.logger.info("LoRA adapter reloaded successfully", lora_id=lora_id)
            return True
            
        except Exception as e:
            self.logger.error("Failed to reload LoRA", lora_id=lora_id, error=str(e))
            return False

    async def remove_lora(self, lora_id: str) -> bool:
        """Remove a LoRA adapter."""
        try:
            if lora_id in self.active_loras:
                del self.active_loras[lora_id]
                self.logger.info("LoRA adapter removed", lora_id=lora_id)
                return True
            else:
                self.logger.warning("LoRA adapter not found", lora_id=lora_id)
                return False
                
        except Exception as e:
            self.logger.error("Failed to remove LoRA", lora_id=lora_id, error=str(e))
            return False

    async def list_loras(self) -> List[str]:
        """List currently loaded LoRA adapters."""
        return list(self.active_loras.keys())

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check and return status."""
        try:
            if not self.model_loaded or self.engine is None:
                return {
                    "status": "unhealthy",
                    "reason": "Engine not initialized",
                    "model_loaded": False
                }
            
            # Quick generation test
            test_start = time.time()
            try:
                test_response = await self.generate(
                    "Hello",
                    max_tokens=5,
                    temperature=0.1
                )
                test_latency = time.time() - test_start
                test_passed = len(test_response) > 0
            except Exception as e:
                test_passed = False
                test_latency = None
                test_response = str(e)
            
            return {
                "status": "healthy" if test_passed else "degraded",
                "model_loaded": self.model_loaded,
                "active_loras": len(self.active_loras),
                "lora_list": list(self.active_loras.keys()),
                "test_generation": {
                    "passed": test_passed,
                    "latency": test_latency,
                    "response": test_response[:100] if test_passed else test_response
                },
                "stats": self.generation_stats.copy()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "reason": f"Health check failed: {str(e)}",
                "model_loaded": self.model_loaded
            }

    async def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics."""
        return {
            "model_name": self.config.model.name,
            "model_dtype": self.config.model.dtype,
            "max_model_len": self.config.model.max_model_len,
            "engine_type": "vllm",
            "lora_enabled": self.config.serving.enable_lora,
            "max_loras": self.config.serving.max_loras,
            "max_lora_rank": self.config.serving.max_lora_rank,
            "active_loras": len(self.active_loras),
            "gpu_memory_utilization": self.config.serving.gpu_memory_utilization,
            "stats": self.generation_stats.copy()
        }

    def _update_stats(self, latency: float, num_tokens: int):
        """Update generation statistics."""
        self.generation_stats["total_requests"] += 1
        self.generation_stats["total_tokens"] += num_tokens
        
        # Update rolling average latency
        total_requests = self.generation_stats["total_requests"]
        current_avg = self.generation_stats["avg_latency"]
        self.generation_stats["avg_latency"] = (
            (current_avg * (total_requests - 1) + latency) / total_requests
        )
        
        self.generation_stats["last_request_time"] = time.time()

    async def shutdown(self):
        """Shutdown the vLLM engine."""
        await super().shutdown()
        if self.engine is not None:
            # vLLM doesn't have explicit shutdown method
            # Let garbage collection handle cleanup
            self.engine = None
        self.active_loras.clear() 