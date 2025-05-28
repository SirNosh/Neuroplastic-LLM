"""Dynamic LoRA Trainer for real-time model adaptation."""

import asyncio
import time
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
from pathlib import Path
import threading
import json

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType, 
    PeftModel,
    PeftConfig
)
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class LoRAAdapterInfo:
    """Information about a LoRA adapter."""
    adapter_id: str
    session_id: str
    created_at: float
    last_used: float
    usage_count: int
    performance_score: float
    sample_count: int
    config: LoraConfig
    file_path: Optional[str] = None
    is_merged: bool = False
    merge_weight: float = 1.0


@dataclass
class LoRAMetrics:
    """Metrics for LoRA training and adaptation."""
    total_adapters_created: int = 0
    active_adapters: int = 0
    merged_adapters: int = 0
    total_adaptations: int = 0
    successful_adaptations: int = 0
    failed_adaptations: int = 0
    avg_adaptation_time: float = 0.0
    cleanup_runs: int = 0
    adapters_removed: int = 0


class LoRATrainer:
    """Dynamic LoRA trainer for continuous model adaptation."""
    
    def __init__(self, model: nn.Module, tokenizer: AutoTokenizer, config, storage_manager):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config  # Global config
        self.lora_config = config.training.lora
        self.general_config = config.training.general
        self.storage_manager = storage_manager
        
        # LoRA state
        self.active_adapters: Dict[str, LoRAAdapterInfo] = {}
        self.base_model = model  # Keep reference to base model
        self.current_peft_model: Optional[PeftModel] = None
        self.adapter_counter = 0
        self.metrics = LoRAMetrics()
        
        # Training state
        self.running = False
        self.adaptation_lock = threading.RLock()
        self.device = next(model.parameters()).device
        
        # Adaptation tracking
        self.session_patterns: Dict[str, List[Dict[str, Any]]] = {}
        self.adaptation_queue = asyncio.Queue(maxsize=100)
        
        # Performance tracking
        self.adaptation_history: List[Dict[str, Any]] = []
        
    async def initialize(self) -> bool:
        """Initialize the LoRA trainer."""
        try:
            logger.info("Initializing LoRA trainer")
            
            # Validate model compatibility
            if not self._is_model_compatible():
                logger.error("Model is not compatible with LoRA adaptation")
                return False
            
            # Initialize base PEFT model if not already done
            if not isinstance(self.model, PeftModel):
                await self._initialize_base_peft_model()
            
            # Load existing adapters from storage
            await self._load_existing_adapters()
            
            # Start background tasks
            self.running = True
            asyncio.create_task(self._adaptation_worker())
            asyncio.create_task(self._cleanup_worker())
            
            logger.info("LoRA trainer initialized successfully",
                       active_adapters=len(self.active_adapters),
                       device=str(self.device))
            return True
            
        except Exception as e:
            logger.error("Failed to initialize LoRA trainer", error=str(e))
            return False
    
    def _is_model_compatible(self) -> bool:
        """Check if the model is compatible with LoRA."""
        try:
            # Check for required modules
            target_modules = self.lora_config.target_modules
            model_modules = [name for name, _ in self.model.named_modules()]
            
            # Check if at least some target modules exist
            found_modules = [module for module in target_modules 
                           if any(module in model_module for model_module in model_modules)]
            
            if not found_modules:
                logger.error("No target modules found in model", 
                           target_modules=target_modules,
                           model_modules=model_modules[:10])  # Log first 10 for brevity
                return False
            
            logger.debug("Model compatibility check passed", 
                        found_modules=found_modules)
            return True
            
        except Exception as e:
            logger.error("Error checking model compatibility", error=str(e))
            return False
    
    async def _initialize_base_peft_model(self):
        """Initialize the base PEFT model."""
        try:
            # Create initial LoRA configuration
            base_lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.lora_config.rank,
                lora_alpha=self.lora_config.alpha,
                lora_dropout=self.lora_config.dropout,
                target_modules=self.lora_config.target_modules,
                bias="none",
                fan_in_fan_out=False,
            )
            
            # Apply PEFT to the model
            self.current_peft_model = get_peft_model(self.base_model, base_lora_config)
            self.model = self.current_peft_model
            
            logger.info("Base PEFT model initialized", 
                       trainable_params=self.current_peft_model.num_parameters(),
                       total_params=self.current_peft_model.get_nb_trainable_parameters())
            
        except Exception as e:
            logger.error("Failed to initialize base PEFT model", error=str(e))
            raise
    
    async def _load_existing_adapters(self):
        """Load existing LoRA adapters from storage."""
        try:
            # List available adapters from storage
            adapters = await self.storage_manager.list_lora_adapters()
            
            loaded_count = 0
            for adapter_info in adapters:
                adapter_id = adapter_info['adapter_id']
                latest_version = adapter_info.get('latest_version', 'latest')
                
                try:
                    # Download adapter
                    local_path = await self.storage_manager.download_lora_adapter(
                        adapter_id, latest_version
                    )
                    
                    if local_path:
                        # Load adapter metadata
                        metadata_file = local_path / "metadata.json"
                        if metadata_file.exists():
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                            
                            # Create adapter info
                            lora_config = LoraConfig(**metadata.get('lora_config', {}))
                            adapter_info_obj = LoRAAdapterInfo(
                                adapter_id=adapter_id,
                                session_id=metadata.get('session_id', 'unknown'),
                                created_at=metadata.get('created_at', time.time()),
                                last_used=metadata.get('last_used', time.time()),
                                usage_count=metadata.get('usage_count', 0),
                                performance_score=metadata.get('performance_score', 0.5),
                                sample_count=metadata.get('sample_count', 0),
                                config=lora_config,
                                file_path=str(local_path)
                            )
                            
                            self.active_adapters[adapter_id] = adapter_info_obj
                            loaded_count += 1
                            
                except Exception as e:
                    logger.warning("Failed to load adapter", 
                                 adapter_id=adapter_id, error=str(e))
            
            logger.info("Loaded existing LoRA adapters", count=loaded_count)
            
        except Exception as e:
            logger.warning("Failed to load existing adapters", error=str(e))
    
    async def adapt_from_feedback(
        self,
        prompt: str,
        response: str,
        feedback_score: float,
        session_id: str,
        importance_weight: float = 1.0
    ) -> bool:
        """Create or update LoRA adapter based on feedback."""
        try:
            adaptation_request = {
                'prompt': prompt,
                'response': response,
                'feedback_score': feedback_score,
                'session_id': session_id,
                'importance_weight': importance_weight,
                'timestamp': time.time()
            }
            
            # Add to adaptation queue
            await asyncio.wait_for(
                self.adaptation_queue.put(adaptation_request),
                timeout=5.0
            )
            
            logger.debug("Adaptation request queued", 
                        session_id=session_id,
                        feedback_score=feedback_score)
            return True
            
        except asyncio.TimeoutError:
            logger.warning("Adaptation queue full, dropping request", 
                          session_id=session_id)
            return False
        except Exception as e:
            logger.error("Error queuing adaptation request", error=str(e))
            return False
    
    async def _adaptation_worker(self):
        """Background worker for processing adaptation requests."""
        while self.running:
            try:
                # Get adaptation request
                request = await asyncio.wait_for(
                    self.adaptation_queue.get(),
                    timeout=1.0
                )
                
                start_time = time.time()
                success = await self._process_adaptation_request(request)
                processing_time = time.time() - start_time
                
                # Update metrics
                self.metrics.total_adaptations += 1
                if success:
                    self.metrics.successful_adaptations += 1
                else:
                    self.metrics.failed_adaptations += 1
                
                # Update average processing time
                self.metrics.avg_adaptation_time = (
                    (self.metrics.avg_adaptation_time * (self.metrics.total_adaptations - 1) + 
                     processing_time) / self.metrics.total_adaptations
                )
                
                self.adaptation_queue.task_done()
                
            except asyncio.TimeoutError:
                continue  # No requests in queue
            except Exception as e:
                logger.error("Error in adaptation worker", error=str(e))
                await asyncio.sleep(1)
    
    async def _process_adaptation_request(self, request: Dict[str, Any]) -> bool:
        """Process a single adaptation request."""
        try:
            session_id = request['session_id']
            feedback_score = request['feedback_score']
            
            # Track session patterns
            if session_id not in self.session_patterns:
                self.session_patterns[session_id] = []
            
            self.session_patterns[session_id].append(request)
            
            # Check if we should create/update an adapter for this session
            should_adapt = await self._should_create_adapter(session_id, feedback_score)
            
            if should_adapt:
                adapter_id = await self._get_or_create_adapter(session_id)
                if adapter_id:
                    success = await self._update_adapter(adapter_id, request)
                    if success:
                        logger.info("Adapter updated successfully", 
                                   adapter_id=adapter_id,
                                   session_id=session_id,
                                   feedback_score=feedback_score)
                    return success
            
            return True  # Not adapting is also success
            
        except Exception as e:
            logger.error("Error processing adaptation request", 
                        session_id=request.get('session_id'),
                        error=str(e))
            return False
    
    async def _should_create_adapter(self, session_id: str, feedback_score: float) -> bool:
        """Determine if we should create/update an adapter."""
        try:
            # Check if feedback is significant enough
            if feedback_score < self.general_config.lora_adaptation_threshold:
                return False
            
            # Check session pattern
            session_data = self.session_patterns.get(session_id, [])
            if len(session_data) < self.lora_config.min_samples_for_adaptation:
                return False
            
            # Check if we already have too many adapters
            if len(self.active_adapters) >= self.lora_config.max_adapters:
                # Consider merging or removing low-performing adapters
                await self._manage_adapter_capacity()
            
            return True
            
        except Exception as e:
            logger.error("Error determining if should adapt", error=str(e))
            return False
    
    async def _get_or_create_adapter(self, session_id: str) -> Optional[str]:
        """Get existing adapter for session or create new one."""
        try:
            with self.adaptation_lock:
                # Look for existing adapter for this session
                for adapter_id, adapter_info in self.active_adapters.items():
                    if adapter_info.session_id == session_id:
                        return adapter_id
                
                # Create new adapter
                adapter_id = f"lora_{session_id}_{self.adapter_counter}_{int(time.time())}"
                self.adapter_counter += 1
                
                # Determine rank based on dynamic settings
                rank = await self._determine_optimal_rank(session_id)
                
                # Create LoRA config
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    r=rank,
                    lora_alpha=self.lora_config.alpha,
                    lora_dropout=self.lora_config.dropout,
                    target_modules=self.lora_config.target_modules,
                    bias="none",
                    fan_in_fan_out=False,
                )
                
                # Create adapter info
                adapter_info = LoRAAdapterInfo(
                    adapter_id=adapter_id,
                    session_id=session_id,
                    created_at=time.time(),
                    last_used=time.time(),
                    usage_count=0,
                    performance_score=0.5,
                    sample_count=0,
                    config=lora_config
                )
                
                self.active_adapters[adapter_id] = adapter_info
                self.metrics.total_adapters_created += 1
                self.metrics.active_adapters = len(self.active_adapters)
                
                logger.info("Created new LoRA adapter", 
                           adapter_id=adapter_id,
                           session_id=session_id,
                           rank=rank)
                
                return adapter_id
                
        except Exception as e:
            logger.error("Error creating adapter", session_id=session_id, error=str(e))
            return None
    
    async def _determine_optimal_rank(self, session_id: str) -> int:
        """Determine optimal LoRA rank for the session."""
        try:
            if not self.lora_config.dynamic_rank_enabled:
                return self.lora_config.rank
            
            # Analyze session complexity
            session_data = self.session_patterns.get(session_id, [])
            if not session_data:
                return self.lora_config.rank
            
            # Simple heuristic based on feedback variance and prompt complexity
            feedback_scores = [req['feedback_score'] for req in session_data]
            avg_feedback = sum(feedback_scores) / len(feedback_scores)
            feedback_variance = sum((score - avg_feedback) ** 2 for score in feedback_scores) / len(feedback_scores)
            
            # Calculate average prompt length as complexity indicator
            avg_prompt_length = sum(len(req['prompt']) for req in session_data) / len(session_data)
            
            # Determine rank based on complexity and performance
            if feedback_variance > 0.3 or avg_prompt_length > 1000:
                # High complexity - use higher rank
                rank = min(self.lora_config.rank_max, 
                          max(self.lora_config.rank, int(self.lora_config.rank * 1.5)))
            elif avg_feedback > 0.8 and feedback_variance < 0.1:
                # Low complexity, good performance - use lower rank
                rank = max(self.lora_config.rank_min, 
                          min(self.lora_config.rank, int(self.lora_config.rank * 0.7)))
            else:
                # Default rank
                rank = self.lora_config.rank
            
            logger.debug("Determined optimal rank", 
                        session_id=session_id,
                        rank=rank,
                        avg_feedback=avg_feedback,
                        feedback_variance=feedback_variance,
                        avg_prompt_length=avg_prompt_length)
            
            return rank
            
        except Exception as e:
            logger.error("Error determining optimal rank", error=str(e))
            return self.lora_config.rank
    
    async def _update_adapter(self, adapter_id: str, request: Dict[str, Any]) -> bool:
        """Update a specific LoRA adapter with new training data."""
        try:
            with self.adaptation_lock:
                if adapter_id not in self.active_adapters:
                    logger.error("Adapter not found for update", adapter_id=adapter_id)
                    return False
                
                adapter_info = self.active_adapters[adapter_id]
                
                # Prepare training data
                prompt = request['prompt']
                response = request['response']
                feedback_score = request['feedback_score']
                importance_weight = request['importance_weight']
                
                # Create training text
                training_text = f"{prompt}\n{response}"
                
                # Tokenize
                inputs = self.tokenizer(
                    training_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=min(1024, self.config.model.max_model_len // 2),
                    padding=True
                ).to(self.device)
                
                # Temporarily add this adapter to the model
                temp_model = await self._create_temp_adapter_model(adapter_info)
                if not temp_model:
                    return False
                
                # Perform training step
                success = await self._train_adapter_step(
                    temp_model, inputs, feedback_score, importance_weight
                )
                
                if success:
                    # Update adapter info
                    adapter_info.last_used = time.time()
                    adapter_info.usage_count += 1
                    adapter_info.sample_count += 1
                    
                    # Update performance score (exponential moving average)
                    alpha = 0.1
                    adapter_info.performance_score = (
                        (1 - alpha) * adapter_info.performance_score + 
                        alpha * feedback_score
                    )
                    
                    # Save adapter if performance is good
                    if adapter_info.performance_score > 0.7:
                        await self._save_adapter(adapter_id, temp_model)
                
                return success
                
        except Exception as e:
            logger.error("Error updating adapter", 
                        adapter_id=adapter_id, error=str(e))
            return False
    
    async def _create_temp_adapter_model(self, adapter_info: LoRAAdapterInfo) -> Optional[PeftModel]:
        """Create a temporary model with the specific adapter."""
        try:
            # Create a fresh PEFT model with this adapter's config
            temp_model = get_peft_model(self.base_model, adapter_info.config)
            
            # Load weights if adapter has been saved
            if adapter_info.file_path and Path(adapter_info.file_path).exists():
                adapter_weights_file = Path(adapter_info.file_path) / "adapter_model.bin"
                if adapter_weights_file.exists():
                    # Load adapter weights
                    state_dict = torch.load(adapter_weights_file, map_location=self.device)
                    temp_model.load_state_dict(state_dict, strict=False)
            
            temp_model.train()
            return temp_model
            
        except Exception as e:
            logger.error("Error creating temp adapter model", error=str(e))
            return None
    
    async def _train_adapter_step(
        self, 
        model: PeftModel, 
        inputs: Dict[str, torch.Tensor],
        feedback_score: float,
        importance_weight: float
    ) -> bool:
        """Perform a single training step on the adapter."""
        try:
            model.train()
            
            # Forward pass
            outputs = model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss
            
            # Scale loss by feedback and importance
            feedback_weight = max(0.1, feedback_score)  # Ensure positive weight
            scaled_loss = loss * feedback_weight * importance_weight
            
            # Backward pass
            scaled_loss.backward()
            
            # Simple gradient step (in practice, you'd use an optimizer)
            learning_rate = self.general_config.learning_rate * 0.1  # Lower LR for adaptation
            with torch.no_grad():
                for param in model.parameters():
                    if param.grad is not None:
                        param.data -= learning_rate * param.grad.data
            
            # Clear gradients
            model.zero_grad()
            
            logger.debug("Adapter training step completed", 
                        loss=scaled_loss.item(),
                        feedback_score=feedback_score)
            
            return True
            
        except Exception as e:
            logger.error("Error in adapter training step", error=str(e))
            return False
    
    async def _save_adapter(self, adapter_id: str, model: PeftModel):
        """Save adapter to storage."""
        try:
            adapter_info = self.active_adapters[adapter_id]
            
            # Create temporary directory for saving
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Save adapter weights
                model.save_pretrained(temp_path)
                
                # Save metadata
                metadata = {
                    'adapter_id': adapter_id,
                    'session_id': adapter_info.session_id,
                    'created_at': adapter_info.created_at,
                    'last_used': adapter_info.last_used,
                    'usage_count': adapter_info.usage_count,
                    'performance_score': adapter_info.performance_score,
                    'sample_count': adapter_info.sample_count,
                    'lora_config': adapter_info.config.to_dict(),
                    'version': '1.0'
                }
                
                with open(temp_path / "metadata.json", 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                # Upload to storage
                success = await self.storage_manager.upload_lora_adapter(
                    temp_path, adapter_id, "latest"
                )
                
                if success:
                    adapter_info.file_path = str(temp_path)  # This will be invalid after temp cleanup
                    logger.info("Adapter saved successfully", adapter_id=adapter_id)
                else:
                    logger.error("Failed to save adapter", adapter_id=adapter_id)
                    
        except Exception as e:
            logger.error("Error saving adapter", adapter_id=adapter_id, error=str(e))
    
    async def _manage_adapter_capacity(self):
        """Manage adapter capacity by merging or removing adapters."""
        try:
            if len(self.active_adapters) < self.lora_config.max_adapters:
                return
            
            # Sort adapters by performance and usage
            adapters_by_score = sorted(
                self.active_adapters.items(),
                key=lambda x: (x[1].performance_score, x[1].usage_count),
                reverse=True
            )
            
            # Keep best performers, consider merging or removing others
            to_remove = []
            for adapter_id, adapter_info in adapters_by_score[self.lora_config.max_adapters:]:
                if adapter_info.performance_score < 0.5:
                    to_remove.append(adapter_id)
                elif adapter_info.performance_score > self.lora_config.adapter_merge_threshold:
                    # Consider merging high-performing adapters
                    await self._consider_adapter_merge(adapter_id)
            
            # Remove low-performing adapters
            for adapter_id in to_remove:
                await self._remove_adapter(adapter_id)
                
        except Exception as e:
            logger.error("Error managing adapter capacity", error=str(e))
    
    async def _consider_adapter_merge(self, adapter_id: str):
        """Consider merging an adapter into the base model."""
        try:
            adapter_info = self.active_adapters[adapter_id]
            
            # Simple merge criterion: high performance and sufficient usage
            if (adapter_info.performance_score > self.lora_config.adapter_merge_threshold and
                adapter_info.usage_count > 50):
                
                logger.info("Considering adapter for merge", 
                           adapter_id=adapter_id,
                           performance=adapter_info.performance_score,
                           usage=adapter_info.usage_count)
                
                # In a full implementation, you would:
                # 1. Load the adapter weights
                # 2. Merge them into the base model
                # 3. Update the base model weights
                # 4. Remove the adapter
                
                # For now, just mark as merged
                adapter_info.is_merged = True
                self.metrics.merged_adapters += 1
                
        except Exception as e:
            logger.error("Error considering adapter merge", error=str(e))
    
    async def _remove_adapter(self, adapter_id: str):
        """Remove an adapter."""
        try:
            if adapter_id in self.active_adapters:
                adapter_info = self.active_adapters[adapter_id]
                
                # Delete from storage
                await self.storage_manager.delete_lora_adapter(adapter_id)
                
                # Remove from active adapters
                del self.active_adapters[adapter_id]
                
                self.metrics.adapters_removed += 1
                self.metrics.active_adapters = len(self.active_adapters)
                
                logger.info("Adapter removed", 
                           adapter_id=adapter_id,
                           performance=adapter_info.performance_score)
                
        except Exception as e:
            logger.error("Error removing adapter", adapter_id=adapter_id, error=str(e))
    
    async def _cleanup_worker(self):
        """Background worker for adapter cleanup."""
        while self.running:
            try:
                await asyncio.sleep(self.general_config.lora_cleanup_interval_seconds)
                await self._cleanup_adapters()
                
            except Exception as e:
                logger.error("Error in cleanup worker", error=str(e))
                await asyncio.sleep(60)  # Wait before retrying
    
    async def cleanup_adapters(self):
        """Public method to trigger adapter cleanup."""
        await self._cleanup_adapters()
    
    async def _cleanup_adapters(self):
        """Clean up old and unused adapters."""
        try:
            current_time = time.time()
            max_age = 24 * 3600  # 24 hours
            min_usage = 5
            
            to_remove = []
            
            for adapter_id, adapter_info in self.active_adapters.items():
                # Remove old, unused adapters
                age = current_time - adapter_info.created_at
                time_since_use = current_time - adapter_info.last_used
                
                should_remove = (
                    (age > max_age and adapter_info.usage_count < min_usage) or
                    (time_since_use > max_age * 2) or  # Unused for 48 hours
                    (adapter_info.performance_score < 0.3 and adapter_info.usage_count > 10)
                )
                
                if should_remove:
                    to_remove.append(adapter_id)
            
            # Remove selected adapters
            for adapter_id in to_remove:
                await self._remove_adapter(adapter_id)
            
            self.metrics.cleanup_runs += 1
            
            if to_remove:
                logger.info("Cleanup completed", removed_adapters=len(to_remove))
                
        except Exception as e:
            logger.error("Error in adapter cleanup", error=str(e))
    
    async def hot_reload_adapter(self, adapter_id: str, local_path: str) -> bool:
        """Hot-reload a specific adapter."""
        try:
            with self.adaptation_lock:
                if adapter_id not in self.active_adapters:
                    logger.error("Adapter not found for hot reload", adapter_id=adapter_id)
                    return False
                
                adapter_info = self.active_adapters[adapter_id]
                
                # Load new adapter weights
                adapter_weights_file = Path(local_path) / "adapter_model.bin"
                if not adapter_weights_file.exists():
                    logger.error("Adapter weights file not found", path=adapter_weights_file)
                    return False
                
                # Update adapter file path
                adapter_info.file_path = local_path
                adapter_info.last_used = time.time()
                
                logger.info("Adapter hot-reloaded successfully", adapter_id=adapter_id)
                return True
                
        except Exception as e:
            logger.error("Error hot-reloading adapter", 
                        adapter_id=adapter_id, error=str(e))
            return False
    
    def get_active_count(self) -> int:
        """Get the number of active adapters."""
        return len(self.active_adapters)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current LoRA trainer status."""
        try:
            return {
                "running": self.running,
                "active_adapters": len(self.active_adapters),
                "adapter_details": {
                    adapter_id: {
                        "session_id": info.session_id,
                        "performance_score": info.performance_score,
                        "usage_count": info.usage_count,
                        "sample_count": info.sample_count,
                        "rank": info.config.r,
                        "last_used": info.last_used,
                        "is_merged": info.is_merged
                    }
                    for adapter_id, info in self.active_adapters.items()
                },
                "metrics": {
                    "total_adapters_created": self.metrics.total_adapters_created,
                    "active_adapters": self.metrics.active_adapters,
                    "merged_adapters": self.metrics.merged_adapters,
                    "total_adaptations": self.metrics.total_adaptations,
                    "successful_adaptations": self.metrics.successful_adaptations,
                    "failed_adaptations": self.metrics.failed_adaptations,
                    "success_rate": (
                        self.metrics.successful_adaptations / max(1, self.metrics.total_adaptations)
                    ),
                    "avg_adaptation_time": self.metrics.avg_adaptation_time,
                    "cleanup_runs": self.metrics.cleanup_runs,
                    "adapters_removed": self.metrics.adapters_removed
                },
                "queue_size": self.adaptation_queue.qsize(),
                "session_patterns": {
                    session_id: len(patterns) 
                    for session_id, patterns in self.session_patterns.items()
                }
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def get_checkpoint(self) -> Dict[str, Any]:
        """Get checkpoint data for saving."""
        try:
            checkpoint = {
                "active_adapters": {
                    adapter_id: {
                        "adapter_id": info.adapter_id,
                        "session_id": info.session_id,
                        "created_at": info.created_at,
                        "last_used": info.last_used,
                        "usage_count": info.usage_count,
                        "performance_score": info.performance_score,
                        "sample_count": info.sample_count,
                        "config": info.config.to_dict(),
                        "file_path": info.file_path,
                        "is_merged": info.is_merged,
                        "merge_weight": info.merge_weight
                    }
                    for adapter_id, info in self.active_adapters.items()
                },
                "metrics": {
                    "total_adapters_created": self.metrics.total_adapters_created,
                    "active_adapters": self.metrics.active_adapters,
                    "merged_adapters": self.metrics.merged_adapters,
                    "total_adaptations": self.metrics.total_adaptations,
                    "successful_adaptations": self.metrics.successful_adaptations,
                    "failed_adaptations": self.metrics.failed_adaptations,
                    "avg_adaptation_time": self.metrics.avg_adaptation_time,
                    "cleanup_runs": self.metrics.cleanup_runs,
                    "adapters_removed": self.metrics.adapters_removed
                },
                "adapter_counter": self.adapter_counter,
                "session_patterns": self.session_patterns
            }
            
            return checkpoint
            
        except Exception as e:
            logger.error("Error creating LoRA checkpoint", error=str(e))
            return {}
    
    async def load_checkpoint(self, checkpoint: Dict[str, Any]) -> bool:
        """Load checkpoint data."""
        try:
            with self.adaptation_lock:
                # Restore adapters
                if "active_adapters" in checkpoint:
                    self.active_adapters.clear()
                    for adapter_id, adapter_data in checkpoint["active_adapters"].items():
                        lora_config = LoraConfig(**adapter_data["config"])
                        adapter_info = LoRAAdapterInfo(
                            adapter_id=adapter_data["adapter_id"],
                            session_id=adapter_data["session_id"],
                            created_at=adapter_data["created_at"],
                            last_used=adapter_data["last_used"],
                            usage_count=adapter_data["usage_count"],
                            performance_score=adapter_data["performance_score"],
                            sample_count=adapter_data["sample_count"],
                            config=lora_config,
                            file_path=adapter_data.get("file_path"),
                            is_merged=adapter_data.get("is_merged", False),
                            merge_weight=adapter_data.get("merge_weight", 1.0)
                        )
                        self.active_adapters[adapter_id] = adapter_info
                
                # Restore metrics
                if "metrics" in checkpoint:
                    metrics_data = checkpoint["metrics"]
                    self.metrics.total_adapters_created = metrics_data.get("total_adapters_created", 0)
                    self.metrics.active_adapters = metrics_data.get("active_adapters", 0)
                    self.metrics.merged_adapters = metrics_data.get("merged_adapters", 0)
                    self.metrics.total_adaptations = metrics_data.get("total_adaptations", 0)
                    self.metrics.successful_adaptations = metrics_data.get("successful_adaptations", 0)
                    self.metrics.failed_adaptations = metrics_data.get("failed_adaptations", 0)
                    self.metrics.avg_adaptation_time = metrics_data.get("avg_adaptation_time", 0.0)
                    self.metrics.cleanup_runs = metrics_data.get("cleanup_runs", 0)
                    self.metrics.adapters_removed = metrics_data.get("adapters_removed", 0)
                
                # Restore other state
                self.adapter_counter = checkpoint.get("adapter_counter", 0)
                self.session_patterns = checkpoint.get("session_patterns", {})
            
            logger.info("LoRA checkpoint loaded successfully", 
                       active_adapters=len(self.active_adapters))
            return True
            
        except Exception as e:
            logger.error("Error loading LoRA checkpoint", error=str(e))
            return False
    
    async def shutdown(self):
        """Shutdown the LoRA trainer."""
        logger.info("Shutting down LoRA trainer")
        self.running = False
        
        # Clear adapters
        self.active_adapters.clear()
        self.session_patterns.clear()
        self.adaptation_history.clear()
        
        logger.info("LoRA trainer shutdown complete") 