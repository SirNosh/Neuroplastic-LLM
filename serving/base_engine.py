"""Base serving engine interface for neuroplastic Qwen system."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import asyncio
import structlog

logger = structlog.get_logger(__name__)


class BaseServingEngine(ABC):
    """Abstract base class for serving engines."""

    def __init__(self, config):
        self.config = config
        self.active_loras: Dict[str, str] = {}
        self.model_loaded = False
        self.logger = logger.bind(engine=self.__class__.__name__)

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the serving engine."""
        pass

    @abstractmethod
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
        """Generate text response."""
        pass

    @abstractmethod
    async def reload_lora(self, lora_path: str, lora_id: str) -> bool:
        """Hot-reload a LoRA adapter."""
        pass

    @abstractmethod
    async def remove_lora(self, lora_id: str) -> bool:
        """Remove a LoRA adapter."""
        pass

    @abstractmethod
    async def list_loras(self) -> List[str]:
        """List currently loaded LoRA adapters."""
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check and return status."""
        pass

    @abstractmethod
    async def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics."""
        pass

    async def shutdown(self):
        """Shutdown the serving engine."""
        self.logger.info("Shutting down serving engine")
        self.model_loaded = False

    def _validate_generate_params(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float
    ) -> bool:
        """Validate generation parameters."""
        if not prompt or not isinstance(prompt, str):
            raise ValueError("Prompt must be a non-empty string")
        
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        
        if not 0.0 <= top_p <= 1.0:
            raise ValueError("top_p must be between 0.0 and 1.0")
        
        return True

    async def warmup(self, sample_prompts: Optional[List[str]] = None):
        """Warm up the model with sample prompts."""
        if not sample_prompts:
            sample_prompts = [
                "Hello, how are you?",
                "What is artificial intelligence?",
                "Explain quantum computing in simple terms."
            ]
        
        self.logger.info("Starting model warmup", num_prompts=len(sample_prompts))
        
        for i, prompt in enumerate(sample_prompts):
            try:
                await self.generate(
                    prompt=prompt,
                    max_tokens=50,
                    temperature=0.1
                )
                self.logger.debug("Warmup completed", prompt_idx=i + 1)
            except Exception as e:
                self.logger.warning("Warmup prompt failed", prompt_idx=i + 1, error=str(e))
        
        self.logger.info("Model warmup completed") 