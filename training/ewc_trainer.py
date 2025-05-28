"""Elastic Weight Consolidation (EWC) trainer for preventing catastrophic forgetting."""

import asyncio
import time
import math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class EWCConfig:
    """Configuration for EWC training."""
    lambda_init: float = 0.4
    lambda_max: float = 10000.0
    lambda_min: float = 0.01
    lambda_decay: float = 0.95
    lambda_adjustment_interval: float = 3600.0  # 1 hour
    fisher_samples: int = 1000
    fisher_batch_size: int = 8
    consolidation_threshold: float = 0.1
    importance_threshold: float = 1e-6
    max_stored_tasks: int = 10


class EWCTrainer:
    """Elastic Weight Consolidation trainer for continuous learning."""
    
    def __init__(self, model: nn.Module, config):
        self.model = model
        self.config = config
        self.ewc_config = config.training.ewc
        
        # EWC state
        self.fisher_matrices: Dict[str, torch.Tensor] = {}
        self.optimal_params: Dict[str, torch.Tensor] = {}
        self.task_lambdas: Dict[str, float] = {}
        self.current_lambda = self.ewc_config.lambda_init
        self.task_counter = 0
        self.total_samples_processed = 0
        
        # Training state
        self.running = False
        self.update_lock = threading.RLock()
        self.device = next(model.parameters()).device
        
        # Performance tracking
        self.last_loss = 0.0
        self.loss_history = []
        self.importance_scores = {}
        
    async def initialize(self) -> bool:
        """Initialize EWC trainer."""
        try:
            logger.info("Initializing EWC trainer")
            
            # Initialize Fisher information matrix for the initial model state
            await self._compute_initial_fisher()
            
            # Store initial optimal parameters
            self._store_optimal_parameters(task_id="initial")
            
            self.running = True
            logger.info("EWC trainer initialized successfully", 
                       lambda_init=self.current_lambda,
                       device=str(self.device))
            return True
            
        except Exception as e:
            logger.error("Failed to initialize EWC trainer", error=str(e))
            return False
    
    async def _compute_initial_fisher(self):
        """Compute initial Fisher information matrix."""
        try:
            logger.info("Computing initial Fisher information matrix")
            
            # Initialize Fisher matrices for all parameters
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.fisher_matrices[name] = torch.zeros_like(param.data)
            
            # Use random samples to estimate Fisher information
            # This part is highly dependent on the actual data distribution and model task.
            # Using a generic approach here for broad applicability.
            if hasattr(self.model, 'config') and hasattr(self.model.config, 'vocab_size'):
                vocab_size = self.model.config.vocab_size
            else:
                # Try to get from tokenizer if available, otherwise default
                tokenizer = getattr(self.model, 'tokenizer', AutoTokenizer.from_pretrained(self.config.model.name, trust_remote_code=self.config.model.trust_remote_code) if hasattr(self.config.model, 'name') else None)
                if tokenizer:
                    vocab_size = tokenizer.vocab_size
                else:
                    vocab_size = 32000 # Fallback vocab size
                    logger.warning(f"Could not determine vocab_size, using default: {vocab_size}")

            batch_size = self.ewc_config.fisher_batch_size
            # Determine seq_len from model config if possible, else default
            if hasattr(self.model, 'config') and hasattr(self.model.config, 'max_position_embeddings'):
                seq_len = min(512, self.model.config.max_position_embeddings) # Cap at 512 for fisher estimation
            else:
                seq_len = 512 # Fallback sequence length
                logger.warning(f"Could not determine max_position_embeddings, using default seq_len: {seq_len} for Fisher estimation")

            with torch.no_grad(): # Ensure no gradients are computed within this estimation block
                self.model.eval() # Set model to evaluation mode
                for _ in range(self.ewc_config.fisher_samples // batch_size): # Adjusted loop count
                    input_ids = torch.randint(
                        0, vocab_size, 
                        (batch_size, seq_len), 
                        device=self.device
                    )
                    attention_mask = torch.ones_like(input_ids)
                    
                    # Forward pass
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    
                    # Compute loss (language modeling loss)
                    if hasattr(outputs, 'logits'):
                        shift_logits = outputs.logits[..., :-1, :].contiguous()
                        shift_labels = input_ids[..., 1:].contiguous()
                        loss = F.cross_entropy(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1),
                            ignore_index=-100
                        )
                    else:
                        # Fallback for models with different output structure
                        loss = outputs.loss if hasattr(outputs, 'loss') else torch.tensor(0.0)
                    
                    # Backward pass to compute gradients
                    loss.backward()
                    
                    # Accumulate Fisher information (square of gradients)
                    for name, param in self.model.named_parameters():
                        if param.grad is not None and name in self.fisher_matrices:
                            self.fisher_matrices[name] += param.grad.data ** 2
            
            # Normalize Fisher matrices
            for name in self.fisher_matrices:
                self.fisher_matrices[name] /= self.ewc_config.fisher_samples
                
            logger.info("Initial Fisher information matrix computed")
            
        except Exception as e:
            logger.error("Failed to compute initial Fisher matrix", error=str(e))
            raise
    
    def _store_optimal_parameters(self, task_id: str):
        """Store current model parameters as optimal for a task."""
        self.optimal_params[task_id] = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.optimal_params[task_id][name] = param.data.clone()
        
        self.task_lambdas[task_id] = self.current_lambda
        logger.debug("Stored optimal parameters", task_id=task_id, lambda_value=self.current_lambda)
    
    async def update_from_sample(
        self, 
        prompt: str, 
        response: str, 
        importance_weight: float = 1.0
    ) -> bool:
        """Update model using EWC regularization from a single sample."""
        try:
            with self.update_lock:
                # Tokenize input
                tokenizer = getattr(self.model, 'tokenizer', None)
                if not tokenizer:
                    logger.warning("No tokenizer available for EWC update")
                    return False
                
                # Prepare input
                text = f"{prompt}\n{response}"
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(self.device)
                
                # Forward pass
                self.model.train()
                outputs = self.model(**inputs, labels=inputs['input_ids'])
                
                # Compute primary loss
                primary_loss = outputs.loss if hasattr(outputs, 'loss') else torch.tensor(0.0)
                
                # Compute EWC regularization loss
                ewc_loss = self._compute_ewc_loss()
                
                # Combined loss
                total_loss = primary_loss + ewc_loss * importance_weight
                
                # Backward pass and optimization
                self.model.zero_grad()
                total_loss.backward()
                
                # Apply gradients (simplified - in practice you'd use an optimizer)
                # This should ideally use the optimizer configured in the main training loop if EWC is part of it.
                # For a standalone EWC, this is a basic update step.
                learning_rate = self.config.training.general.learning_rate # Use general LR
                with torch.no_grad():
                    for param in self.model.parameters():
                        if param.grad is not None:
                            param.data -= learning_rate * param.grad.data
                
                # Update Fisher information incrementally
                await self._update_fisher_incremental(inputs)
                
                # Track metrics
                self.last_loss = total_loss.item()
                self.loss_history.append(self.last_loss)
                if len(self.loss_history) > 1000:
                    self.loss_history = self.loss_history[-1000:]
                
                self.total_samples_processed += 1
                
                logger.debug("EWC update completed", 
                           primary_loss=primary_loss.item(),
                           ewc_loss=ewc_loss.item(),
                           total_loss=total_loss.item(),
                           importance_weight=importance_weight)
                
                return True
                
        except Exception as e:
            logger.error("Error in EWC update", error=str(e))
            return False
    
    def _compute_ewc_loss(self) -> torch.Tensor:
        """Compute EWC regularization loss."""
        ewc_loss = torch.tensor(0.0, device=self.device)
        
        try:
            for task_id, optimal_params in self.optimal_params.items():
                task_lambda = self.task_lambdas.get(task_id, self.current_lambda)
                
                for name, param in self.model.named_parameters():
                    if (param.requires_grad and 
                        name in optimal_params and 
                        name in self.fisher_matrices):
                        
                        # EWC penalty: λ * F * (θ - θ*)²
                        diff = param - optimal_params[name]
                        penalty = self.fisher_matrices[name] * (diff ** 2)
                        ewc_loss += task_lambda * penalty.sum()
            
            return ewc_loss
            
        except Exception as e:
            logger.error("Error computing EWC loss", error=str(e))
            return ewc_loss
    
    async def _update_fisher_incremental(self, inputs: Dict[str, torch.Tensor]):
        """Update Fisher information matrix incrementally."""
        try:
            # Compute gradients for Fisher update
            self.model.zero_grad()
            outputs = self.model(**inputs)
            
            if hasattr(outputs, 'logits'):
                # Sample from the model's predictive distribution
                probs = F.softmax(outputs.logits, dim=-1)
                samples = torch.multinomial(probs.view(-1, probs.size(-1)), 1)
                sampled_logits = outputs.logits.view(-1, outputs.logits.size(-1))
                
                # Compute log-likelihood of sampled tokens
                log_likelihood = F.log_softmax(sampled_logits, dim=-1)
                selected_log_likelihood = log_likelihood.gather(1, samples)
                loss = -selected_log_likelihood.mean()
                
                loss.backward()
                
                # Update Fisher matrices with exponential moving average
                alpha = 0.1  # Update rate
                for name, param in self.model.named_parameters():
                    if param.grad is not None and name in self.fisher_matrices:
                        new_fisher = param.grad.data ** 2
                        self.fisher_matrices[name] = (
                            (1 - alpha) * self.fisher_matrices[name] + 
                            alpha * new_fisher
                        )
                        
        except Exception as e:
            logger.debug("Error updating Fisher matrix incrementally", error=str(e))
    
    async def consolidate_task(self, task_id: Optional[str] = None) -> bool:
        """Consolidate current knowledge as a new task."""
        try:
            with self.update_lock:
                if task_id is None:
                    task_id = f"task_{self.task_counter}"
                    self.task_counter += 1
                
                logger.info("Consolidating task", task_id=task_id)
                
                # Store current parameters as optimal for this task
                self._store_optimal_parameters(task_id)
                
                # Cleanup old tasks if we have too many
                if len(self.optimal_params) > self.ewc_config.max_stored_tasks:
                    await self._cleanup_old_tasks()
                
                logger.info("Task consolidated successfully", 
                           task_id=task_id,
                           total_tasks=len(self.optimal_params))
                return True
                
        except Exception as e:
            logger.error("Error consolidating task", task_id=task_id, error=str(e))
            return False
    
    async def _cleanup_old_tasks(self):
        """Remove old tasks to prevent memory growth."""
        try:
            # Remove oldest tasks, keeping the most recent ones
            task_ids = list(self.optimal_params.keys())
            tasks_to_remove = task_ids[:-self.ewc_config.max_stored_tasks]
            
            for task_id in tasks_to_remove:
                if task_id != "initial":  # Never remove initial task
                    del self.optimal_params[task_id]
                    if task_id in self.task_lambdas:
                        del self.task_lambdas[task_id]
                    logger.debug("Removed old task", task_id=task_id)
                    
        except Exception as e:
            logger.error("Error cleaning up old tasks", error=str(e))
    
    async def adjust_lambda(self) -> float:
        """Adjust lambda based on performance and forgetting detection."""
        try:
            with self.update_lock:
                old_lambda = self.current_lambda
                
                # Calculate recent loss trend
                if len(self.loss_history) >= 10:
                    recent_losses = self.loss_history[-10:]
                    loss_trend = sum(recent_losses) / len(recent_losses)
                    
                    # Increase lambda if loss is increasing (potential forgetting)
                    if len(self.loss_history) >= 20:
                        older_losses = self.loss_history[-20:-10]
                        older_trend = sum(older_losses) / len(older_losses)
                        
                        if loss_trend > older_trend * 1.1:  # 10% increase
                            self.current_lambda = min(
                                self.current_lambda * 1.1,
                                self.ewc_config.lambda_max
                            )
                        elif loss_trend < older_trend * 0.9:  # 10% decrease
                            self.current_lambda = max(
                                self.current_lambda * self.ewc_config.lambda_decay,
                                self.ewc_config.lambda_min
                            )
                
                logger.debug("Lambda adjusted", 
                           old_lambda=old_lambda,
                           new_lambda=self.current_lambda,
                           loss_trend=len(self.loss_history))
                
                return self.current_lambda
                
        except Exception as e:
            logger.error("Error adjusting lambda", error=str(e))
            return self.current_lambda
    
    async def compute_importance_scores(self) -> Dict[str, float]:
        """Compute importance scores for different parameters."""
        try:
            importance_scores = {}
            
            for name, fisher in self.fisher_matrices.items():
                # Compute mean importance across the parameter tensor
                importance = fisher.mean().item()
                importance_scores[name] = importance
                
                # Store for tracking
                self.importance_scores[name] = importance
            
            return importance_scores
            
        except Exception as e:
            logger.error("Error computing importance scores", error=str(e))
            return {}
    
    def get_lambda(self) -> float:
        """Get current lambda value."""
        return self.current_lambda
    
    def get_status(self) -> Dict[str, Any]:
        """Get current EWC trainer status."""
        try:
            return {
                "running": self.running,
                "current_lambda": self.current_lambda,
                "total_tasks": len(self.optimal_params),
                "samples_processed": self.total_samples_processed,
                "last_loss": self.last_loss,
                "avg_recent_loss": (
                    sum(self.loss_history[-10:]) / min(10, len(self.loss_history))
                    if self.loss_history else 0.0
                ),
                "fisher_matrices_count": len(self.fisher_matrices),
                "task_lambdas": dict(self.task_lambdas)
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def get_checkpoint(self) -> Dict[str, Any]:
        """Get checkpoint data for saving."""
        try:
            checkpoint = {
                "fisher_matrices": {
                    name: matrix.cpu() for name, matrix in self.fisher_matrices.items()
                },
                "optimal_params": {
                    task_id: {
                        name: param.cpu() for name, param in params.items()
                    }
                    for task_id, params in self.optimal_params.items()
                },
                "task_lambdas": dict(self.task_lambdas),
                "current_lambda": self.current_lambda,
                "task_counter": self.task_counter,
                "total_samples_processed": self.total_samples_processed,
                "loss_history": self.loss_history[-100:],  # Save recent history
                "importance_scores": dict(self.importance_scores)
            }
            
            return checkpoint
            
        except Exception as e:
            logger.error("Error creating checkpoint", error=str(e))
            return {}
    
    async def load_checkpoint(self, checkpoint: Dict[str, Any]) -> bool:
        """Load checkpoint data."""
        try:
            with self.update_lock:
                # Restore Fisher matrices
                if "fisher_matrices" in checkpoint:
                    self.fisher_matrices = {
                        name: matrix.to(self.device)
                        for name, matrix in checkpoint["fisher_matrices"].items()
                    }
                
                # Restore optimal parameters
                if "optimal_params" in checkpoint:
                    self.optimal_params = {
                        task_id: {
                            name: param.to(self.device)
                            for name, param in params.items()
                        }
                        for task_id, params in checkpoint["optimal_params"].items()
                    }
                
                # Restore other state
                self.task_lambdas = checkpoint.get("task_lambdas", {})
                self.current_lambda = checkpoint.get("current_lambda", self.ewc_config.lambda_init)
                self.task_counter = checkpoint.get("task_counter", 0)
                self.total_samples_processed = checkpoint.get("total_samples_processed", 0)
                self.loss_history = checkpoint.get("loss_history", [])
                self.importance_scores = checkpoint.get("importance_scores", {})
                
            logger.info("EWC checkpoint loaded successfully")
            return True
            
        except Exception as e:
            logger.error("Error loading checkpoint", error=str(e))
            return False
    
    async def shutdown(self):
        """Shutdown EWC trainer."""
        logger.info("Shutting down EWC trainer")
        self.running = False
        
        # Clear memory
        self.fisher_matrices.clear()
        self.optimal_params.clear()
        self.task_lambdas.clear()
        self.loss_history.clear()
        self.importance_scores.clear()
        
        logger.info("EWC trainer shutdown complete") 