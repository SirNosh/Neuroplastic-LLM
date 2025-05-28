"""Pydantic models for API requests and responses."""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator
import time


class GenerateRequest(BaseModel):
    """Request model for text generation."""
    
    prompt: str = Field(..., description="Input prompt for generation", min_length=1)
    max_tokens: int = Field(default=512, description="Maximum tokens to generate", ge=1, le=4096)
    temperature: float = Field(default=0.7, description="Sampling temperature", ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, description="Nucleus sampling parameter", ge=0.0, le=1.0)
    stop_sequences: Optional[List[str]] = Field(default=None, description="Stop sequences")
    lora_id: Optional[str] = Field(default=None, description="LoRA adapter ID to use")
    stream: bool = Field(default=False, description="Whether to stream the response")
    
    @validator('prompt')
    def validate_prompt(cls, v):
        if not v.strip():
            raise ValueError("Prompt cannot be empty or whitespace only")
        return v.strip()
    
    @validator('stop_sequences')
    def validate_stop_sequences(cls, v):
        if v is not None and len(v) > 10:
            raise ValueError("Maximum 10 stop sequences allowed")
        return v


class GenerateResponse(BaseModel):
    """Response model for text generation."""
    
    response: str = Field(..., description="Generated text response")
    prompt: str = Field(..., description="Original input prompt")
    model: str = Field(..., description="Model used for generation")
    lora_id: Optional[str] = Field(default=None, description="LoRA adapter used")
    stats: Dict[str, Any] = Field(..., description="Generation statistics")
    timestamp: float = Field(default_factory=time.time, description="Response timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "response": "Hello! How can I help you today?",
                "prompt": "Hello",
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "lora_id": None,
                "stats": {
                    "latency": 0.15,
                    "tokens_generated": 8,
                    "tokens_per_second": 53.3
                },
                "timestamp": 1703123456.789
            }
        }


class LoRARequest(BaseModel):
    """Request model for LoRA operations."""
    
    lora_id: str = Field(..., description="Unique identifier for the LoRA adapter")
    lora_path: str = Field(..., description="Path to the LoRA adapter files")
    
    @validator('lora_id')
    def validate_lora_id(cls, v):
        if not v.strip():
            raise ValueError("LoRA ID cannot be empty")
        # Simple validation for safe IDs
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError("LoRA ID must contain only alphanumeric characters, hyphens, and underscores")
        return v.strip()


class LoRAResponse(BaseModel):
    """Response model for LoRA operations."""
    
    status: str = Field(..., description="Operation status")
    message: str = Field(..., description="Status message")
    lora_id: str = Field(..., description="LoRA adapter ID")
    timestamp: float = Field(default_factory=time.time, description="Operation timestamp")


class HealthResponse(BaseModel):
    """Response model for health checks."""
    
    status: str = Field(..., description="Overall health status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    active_loras: int = Field(..., description="Number of active LoRA adapters")
    lora_list: List[str] = Field(..., description="List of active LoRA IDs")
    test_generation: Dict[str, Any] = Field(..., description="Test generation results")
    stats: Dict[str, Any] = Field(..., description="System statistics")
    timestamp: float = Field(default_factory=time.time, description="Health check timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "active_loras": 2,
                "lora_list": ["ewc_adapter_001", "ppo_adapter_002"],
                "test_generation": {
                    "passed": True,
                    "latency": 0.08,
                    "response": "Hello!"
                },
                "stats": {
                    "total_requests": 1247,
                    "total_tokens": 15892,
                    "avg_latency": 0.12
                },
                "timestamp": 1703123456.789
            }
        }


class ModelInfoResponse(BaseModel):
    """Response model for model information."""
    
    model_name: str = Field(..., description="Model name")
    model_dtype: str = Field(..., description="Model data type")
    max_model_len: int = Field(..., description="Maximum model length")
    engine_type: str = Field(..., description="Serving engine type")
    lora_enabled: bool = Field(..., description="Whether LoRA is enabled")
    max_loras: int = Field(..., description="Maximum number of LoRA adapters")
    max_lora_rank: int = Field(..., description="Maximum LoRA rank")
    active_loras: int = Field(..., description="Currently active LoRA adapters")
    gpu_memory_utilization: float = Field(..., description="GPU memory utilization")
    stats: Dict[str, Any] = Field(..., description="Runtime statistics")
    timestamp: float = Field(default_factory=time.time, description="Info timestamp")


class ErrorResponse(BaseModel):
    """Response model for errors."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
    timestamp: float = Field(default_factory=time.time, description="Error timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid request parameters",
                "details": {
                    "field": "temperature",
                    "issue": "Value must be between 0.0 and 2.0"
                },
                "timestamp": 1703123456.789
            }
        }


class TraceEvent(BaseModel):
    """Model for trace events sent to Kafka."""
    
    session_id: str = Field(..., description="Session identifier")
    request_id: str = Field(..., description="Request identifier")
    prompt: str = Field(..., description="Input prompt")
    response: str = Field(..., description="Generated response")
    model: str = Field(..., description="Model used")
    lora_id: Optional[str] = Field(default=None, description="LoRA adapter used")
    latency: float = Field(..., description="Generation latency in seconds")
    tokens_generated: int = Field(..., description="Number of tokens generated")
    timestamp: float = Field(default_factory=time.time, description="Event timestamp")
    user_agent: Optional[str] = Field(default=None, description="User agent string")
    ip_address: Optional[str] = Field(default=None, description="Client IP address")
    
    class Config:
        schema_extra = {
            "example": {
                "session_id": "sess_abc123",
                "request_id": "req_1703123456789",
                "prompt": "What is machine learning?",
                "response": "Machine learning is a subset of artificial intelligence...",
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "lora_id": "domain_adapter_v1",
                "latency": 0.235,
                "tokens_generated": 45,
                "timestamp": 1703123456.789,
                "user_agent": "NeuroplasticClient/1.0",
                "ip_address": "192.168.1.100"
            }
        } 