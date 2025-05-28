"""Online trainer for continuous learning in neuroplastic Qwen system."""

import asyncio
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading

import torch
import torch.nn as nn
from transformers import AutoTokenizer
import structlog

from .ewc_trainer import EWCTrainer
from .lora_trainer import LoRATrainer
from .tot_optimizer import ToTOptimizer
from .replay_buffer import ReplayBuffer

logger = structlog.get_logger(__name__)


@dataclass
class TrainingRequest:
    """Represents a training request from user feedback."""
    request_id: str
    session_id: str
    prompt: str
    response: str
    feedback_score: float
    feedback_text: Optional[str] = None
    importance_weight: float = 1.0
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class TrainingMetrics:
    """Training metrics and statistics."""
    total_requests: int = 0
    processed_requests: int = 0
    failed_requests: int = 0
    avg_processing_time: float = 0.0
    ewc_updates: int = 0
    lora_updates: int = 0
    tot_optimizations: int = 0
    current_ewc_lambda: float = 0.0
    active_lora_adapters: int = 0
    replay_buffer_size: int = 0


class OnlineTrainer:
    """Orchestrates continuous learning with multiple training strategies."""
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: AutoTokenizer,
        config,
        kafka_manager,
        storage_manager
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config # Global config
        self.training_config = config.training # TrainingConfig (includes general, ewc, etc.)
        self.kafka_manager = kafka_manager
        self.storage_manager = storage_manager
        
        # Initialize training components
        self.ewc_trainer = EWCTrainer(model, config) # Pass global config
        self.lora_trainer = LoRATrainer(model, tokenizer, config, storage_manager)
        self.tot_optimizer = ToTOptimizer(model, tokenizer, config)
        self.replay_buffer = ReplayBuffer(config)
        
        # Training state
        self.training_queue = asyncio.Queue(maxsize=self.training_config.general.max_queue_size)
        self.metrics = TrainingMetrics()
        self.running = False
        self.training_lock = threading.RLock() # Use threading.RLock for reentrant lock
        self.executor = ThreadPoolExecutor(max_workers=self.training_config.general.max_workers)
        
        # Feedback thresholds
        self.positive_threshold = self.training_config.general.positive_feedback_threshold
        self.negative_threshold = self.training_config.general.negative_feedback_threshold
        self.lora_adaptation_threshold = self.training_config.general.lora_adaptation_threshold
        
    async def initialize(self) -> bool:
        """Initialize the online trainer."""
        try:
            logger.info("Initializing online trainer")
            
            # Initialize components
            if not await self.ewc_trainer.initialize():
                logger.error("Failed to initialize EWC trainer")
                return False
                
            if not await self.lora_trainer.initialize():
                logger.error("Failed to initialize LoRA trainer")
                return False
                
            if not await self.tot_optimizer.initialize():
                logger.error("Failed to initialize ToT optimizer")
                return False
                
            if not await self.replay_buffer.initialize(self.storage_manager):
                logger.error("Failed to initialize replay buffer")
                return False
            
            # Start training workers
            self.running = True
            asyncio.create_task(self._training_worker())
            asyncio.create_task(self._feedback_consumer())
            asyncio.create_task(self._metrics_logger())
            
            logger.info("Online trainer initialized successfully")
            return True
            
        except Exception as e:
            logger.error("Failed to initialize online trainer", error=str(e))
            return False
    
    async def _feedback_consumer(self):
        """Consume feedback messages from Kafka."""
        try:
            async def process_feedback(message: Dict[str, Any]):
                """Process a feedback message."""
                try:
                    request = TrainingRequest(
                        request_id=message['request_id'],
                        session_id=message['session_id'],
                        prompt=message['prompt'],
                        response=message['response'],
                        feedback_score=message['feedback_score'],
                        feedback_text=message.get('feedback_text'),
                        importance_weight=message.get('importance_weight', 1.0),
                        timestamp=message.get('timestamp', time.time())
                    )
                    
                    await self.add_training_request(request)
                    
                except Exception as e:
                    logger.error("Error processing feedback message", error=str(e))
            
            # Start consuming feedback messages
            await self.kafka_manager.consume_messages(
                topic=self.kafka_manager.kafka_config.topics.feedback,
                message_handler=process_feedback,
                consumer_group="online_trainer"
            )
            
        except Exception as e:
            logger.error("Error in feedback consumer", error=str(e))
    
    async def add_training_request(self, request: TrainingRequest):
        """Add a training request to the queue."""
        try:
            if not self.running:
                logger.warning("Online trainer not running")
                return False
            
            # Add to queue with timeout
            await asyncio.wait_for(
                self.training_queue.put(request),
                timeout=self.training_config.general.queue_timeout_seconds # Use correct config field
            )
            
            self.metrics.total_requests += 1
            logger.debug("Training request added", request_id=request.request_id)
            return True
            
        except asyncio.TimeoutError:
            logger.warning("Training queue full, dropping request", request_id=request.request_id)
            return False
        except Exception as e:
            logger.error("Error adding training request", error=str(e))
            return False
    
    async def _training_worker(self):
        """Main training worker that processes requests."""
        while self.running:
            try:
                # Get training request
                request = await asyncio.wait_for(
                    self.training_queue.get(),
                    timeout=1.0
                )
                
                start_time = time.time()
                success = await self._process_training_request(request)
                processing_time = time.time() - start_time
                
                # Update metrics
                if success:
                    self.metrics.processed_requests += 1
                else:
                    self.metrics.failed_requests += 1
                
                # Update average processing time
                total_processed = self.metrics.processed_requests + self.metrics.failed_requests
                self.metrics.avg_processing_time = (
                    (self.metrics.avg_processing_time * (total_processed - 1) + processing_time) / 
                    total_processed
                )
                
                self.training_queue.task_done()
                
            except asyncio.TimeoutError:
                continue  # No requests in queue
            except Exception as e:
                logger.error("Error in training worker", error=str(e))
                await asyncio.sleep(1)
    
    async def _process_training_request(self, request: TrainingRequest) -> bool:
        """Process a single training request."""
        try:
            logger.info("Processing training request", 
                       request_id=request.request_id,
                       feedback_score=request.feedback_score)
            
            # Add to replay buffer
            await self.replay_buffer.add_sample(
                prompt=request.prompt,
                response=request.response,
                feedback_score=request.feedback_score,
                importance_weight=request.importance_weight,
                metadata={
                    'request_id': request.request_id,
                    'session_id': request.session_id,
                    'timestamp': request.timestamp
                }
            )
            
            # Determine training strategy based on feedback
            if request.feedback_score >= self.positive_threshold:
                await self._handle_positive_feedback(request)
            elif request.feedback_score <= self.negative_threshold:
                await self._handle_negative_feedback(request)
            else:
                # Neutral feedback - just add to replay buffer
                logger.debug("Neutral feedback, added to replay buffer", 
                           request_id=request.request_id)
            
            # Periodic optimization tasks
            await self._maybe_run_periodic_tasks()
            
            return True
            
        except Exception as e:
            logger.error("Error processing training request", 
                        request_id=request.request_id, 
                        error=str(e))
            return False
    
    async def _handle_positive_feedback(self, request: TrainingRequest):
        """Handle positive feedback by reinforcing the behavior."""
        try:
            # Use EWC to reinforce positive patterns while preserving existing knowledge
            await self.ewc_trainer.update_from_sample(
                prompt=request.prompt,
                response=request.response,
                importance_weight=request.importance_weight * request.feedback_score
            )
            self.metrics.ewc_updates += 1
            
            # Consider creating a specialized LoRA adapter for this type of interaction
            if request.feedback_score >= self.lora_adaptation_threshold:
                await self.lora_trainer.adapt_from_feedback(
                    prompt=request.prompt,
                    response=request.response,
                    feedback_score=request.feedback_score,
                    session_id=request.session_id
                )
                self.metrics.lora_updates += 1
            
            logger.debug("Processed positive feedback", 
                        request_id=request.request_id,
                        feedback_score=request.feedback_score)
            
        except Exception as e:
            logger.error("Error handling positive feedback", 
                        request_id=request.request_id, 
                        error=str(e))
    
    async def _handle_negative_feedback(self, request: TrainingRequest):
        """Handle negative feedback by learning to avoid similar outputs."""
        try:
            # Use Tree-of-Thought optimization to find better responses
            improved_response = await self.tot_optimizer.optimize_response(
                prompt=request.prompt,
                bad_response=request.response,
                feedback_score=request.feedback_score
            )
            
            if improved_response:
                # Update model with the improved response
                await self.ewc_trainer.update_from_sample(
                    prompt=request.prompt,
                    response=improved_response,
                    importance_weight=request.importance_weight * abs(request.feedback_score)
                )
                
                # Add improved sample to replay buffer
                await self.replay_buffer.add_sample(
                    prompt=request.prompt,
                    response=improved_response,
                    feedback_score=1.0,  # Treat improved response as positive
                    importance_weight=request.importance_weight,
                    metadata={
                        'request_id': f"{request.request_id}_improved",
                        'session_id': request.session_id,
                        'timestamp': time.time(),
                        'original_request_id': request.request_id
                    }
                )
                self.metrics.tot_optimizations += 1
            else:
                # If ToT doesn't find an improvement, or is not used,
                # we might still want to discourage the bad response via EWC.
                # This is a design choice. For now, EWC is mainly for positive reinforcement.
                logger.debug("No improved response from ToT or ToT disabled for negative feedback",
                               request_id=request.request_id)

            logger.debug("Processed negative feedback", 
                        request_id=request.request_id,
                        feedback_score=request.feedback_score,
                        improved=improved_response is not None)
            
        except Exception as e:
            logger.error("Error handling negative feedback", 
                        request_id=request.request_id, 
                        error=str(e))
    
    async def _maybe_run_periodic_tasks(self):
        """Run periodic maintenance and optimization tasks."""
        try:
            current_time = time.time()
            
            # Replay buffer optimization
            if (hasattr(self, '_last_replay_optimization') and
                current_time - self._last_replay_optimization > 
                self.training_config.general.replay_optimization_interval_seconds):
                await self.replay_buffer.optimize()
                self._last_replay_optimization = current_time
            
            # EWC lambda adjustment
            if (hasattr(self, '_last_ewc_adjustment') and
                current_time - self._last_ewc_adjustment > 
                self.training_config.general.ewc_adjustment_interval_seconds):
                await self.ewc_trainer.adjust_lambda()
                self.metrics.current_ewc_lambda = self.ewc_trainer.get_lambda()
                self._last_ewc_adjustment = current_time
            
            # LoRA adapter cleanup
            if (hasattr(self, '_last_lora_cleanup') and
                current_time - self._last_lora_cleanup > 
                self.training_config.general.lora_cleanup_interval_seconds):
                await self.lora_trainer.cleanup_adapters()
                self.metrics.active_lora_adapters = self.lora_trainer.get_active_count()
                self._last_lora_cleanup = current_time
            
            # Initialize timestamps if not set
            if not hasattr(self, '_last_replay_optimization'): self._last_replay_optimization = current_time
            if not hasattr(self, '_last_ewc_adjustment'): self._last_ewc_adjustment = current_time
            if not hasattr(self, '_last_lora_cleanup'): self._last_lora_cleanup = current_time
                
        except Exception as e:
            logger.error("Error in periodic tasks", error=str(e))
    
    async def _metrics_logger(self):
        """Periodically log training metrics."""
        while self.running:
            try:
                await asyncio.sleep(self.training_config.general.metrics_log_interval_seconds)
                
                # Update dynamic metrics
                self.metrics.replay_buffer_size = await self.replay_buffer.get_size()
                self.metrics.active_lora_adapters = self.lora_trainer.get_active_count()
                self.metrics.current_ewc_lambda = self.ewc_trainer.get_lambda()
                
                logger.info("Training metrics", **self.metrics.__dict__)
                
                # Send metrics to Kafka for monitoring
                await self.kafka_manager.send_metrics({
                    "component": "online_trainer",
                    **self.metrics.__dict__
                })
                
            except Exception as e:
                logger.error("Error logging metrics", error=str(e))
    
    async def force_training_cycle(self, num_samples: int = 10) -> Dict[str, Any]:
        """Force a training cycle using samples from replay buffer."""
        try:
            with self.training_lock:
                logger.info("Starting forced training cycle", num_samples=num_samples)
                
                # Get samples from replay buffer
                samples = await self.replay_buffer.get_batch(num_samples)
                
                if not samples:
                    logger.warning("No samples available for forced training cycle (replay buffer empty or not implemented)")
                    return {"message": "No samples available in replay buffer for forced training cycle"}
                
                # Process samples through all training components
                results = {
                    "samples_processed": len(samples),
                    "ewc_updates": 0,
                    "lora_updates": 0,
                    "tot_optimizations": 0
                }
                
                for sample in samples:
                    # EWC update
                    await self.ewc_trainer.update_from_sample(
                        prompt=sample['prompt'],
                        response=sample['response'],
                        importance_weight=sample.get('importance_weight', 1.0)
                    )
                    results["ewc_updates"] += 1
                    
                    # LoRA adaptation if high feedback score
                    if sample.get('feedback_score', 0) >= self.training_config.general.lora_adaptation_threshold:
                        await self.lora_trainer.adapt_from_feedback(
                            prompt=sample['prompt'],
                            response=sample['response'],
                            feedback_score=sample['feedback_score'],
                            session_id=sample.get('metadata', {}).get('session_id', 'forced')
                        )
                        results["lora_updates"] += 1
                
                logger.info("Forced training cycle completed", **results)
                return results
                
        except Exception as e:
            logger.error("Error in forced training cycle", error=str(e))
            return {"error": str(e)}
    
    async def get_training_status(self) -> Dict[str, Any]:
        """Get current training status and metrics."""
        try:
            # Update dynamic metrics
            self.metrics.replay_buffer_size = await self.replay_buffer.get_size()
            self.metrics.active_lora_adapters = self.lora_trainer.get_active_count()
            if self.ewc_trainer.running: # Check if EWC trainer is running
                 self.metrics.current_ewc_lambda = self.ewc_trainer.get_lambda()
            
            return {
                "running": self.running,
                "queue_size": self.training_queue.qsize(),
                "max_queue_size": self.training_config.general.max_queue_size,
                "metrics": self.metrics.__dict__,
                "component_status": {
                    "ewc_trainer": self.ewc_trainer.get_status(),
                    "lora_trainer": self.lora_trainer.get_status(),
                    "tot_optimizer": self.tot_optimizer.get_status(),
                    "replay_buffer": await self.replay_buffer.get_status()
                }
            }
            
        except Exception as e:
            logger.error("Error getting training status", error=str(e))
            return {"error": str(e)}
    
    async def save_checkpoint(self, checkpoint_id: str) -> bool:
        """Save a training checkpoint."""
        try:
            checkpoint_data = {
                "metrics": self.metrics.__dict__,
                "ewc_state": await self.ewc_trainer.get_checkpoint(),
                "lora_state": await self.lora_trainer.get_checkpoint(),
                "tot_state": await self.tot_optimizer.get_checkpoint(),
                "replay_buffer_state": await self.replay_buffer.get_checkpoint()
            }
            
            return await self.storage_manager.save_checkpoint(
                model_state=checkpoint_data,
                checkpoint_id=checkpoint_id,
                metadata={
                    "component": "online_trainer",
                    "timestamp": time.time(),
                    "metrics": self.metrics.__dict__
                }
            )
            
        except Exception as e:
            logger.error("Error saving checkpoint", error=str(e))
            return False
    
    async def load_checkpoint(self, checkpoint_id: str) -> bool:
        """Load a training checkpoint."""
        try:
            checkpoint_data = await self.storage_manager.load_checkpoint(checkpoint_id)
            
            if not checkpoint_data:
                return False
            
            # Restore component states
            if "ewc_state" in checkpoint_data:
                await self.ewc_trainer.load_checkpoint(checkpoint_data["ewc_state"])
            
            if "lora_state" in checkpoint_data:
                await self.lora_trainer.load_checkpoint(checkpoint_data["lora_state"])
            
            if "tot_state" in checkpoint_data:
                await self.tot_optimizer.load_checkpoint(checkpoint_data["tot_state"])
            
            if "replay_buffer_state" in checkpoint_data:
                await self.replay_buffer.load_checkpoint(checkpoint_data["replay_buffer_state"])
            
            # Restore metrics
            if "metrics" in checkpoint_data:
                for key, value in checkpoint_data["metrics"].items():
                    if hasattr(self.metrics, key):
                        setattr(self.metrics, key, value)
            
            logger.info("Checkpoint loaded successfully", checkpoint_id=checkpoint_id)
            return True
            
        except Exception as e:
            logger.error("Error loading checkpoint", checkpoint_id=checkpoint_id, error=str(e))
            return False
    
    async def shutdown(self):
        """Shutdown the online trainer."""
        logger.info("Shutting down online trainer")
        self.running = False
        
        # Shutdown components
        if hasattr(self, 'ewc_trainer') and self.ewc_trainer: await self.ewc_trainer.shutdown()
        if hasattr(self, 'lora_trainer') and self.lora_trainer: await self.lora_trainer.shutdown()
        if hasattr(self, 'tot_optimizer') and self.tot_optimizer: await self.tot_optimizer.shutdown()
        if hasattr(self, 'replay_buffer') and self.replay_buffer: await self.replay_buffer.shutdown()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("Online trainer shutdown complete") 