"""Prioritized Replay Buffer for continuous learning.

Implements experience replay with prioritized sampling for the neuroplastic Qwen system.
"""

import asyncio
import time
import random
import heapq
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ReplayItem:
    """A single item in the replay buffer with priority metadata."""
    prompt: str
    response: str
    feedback_score: float
    importance_weight: float
    priority: float  # For prioritized sampling
    timestamp: float
    last_sampled: Optional[float]  # Time when this was last sampled
    sampling_count: int  # Number of times this item has been sampled
    metadata: Dict[str, Any]
    
    @property
    def age(self) -> float:
        """Get the age of this item in seconds."""
        return time.time() - self.timestamp
    
    @property
    def staleness(self) -> float:
        """Get the staleness (time since last sampled) in seconds."""
        if self.last_sampled is None:
            return float('inf')  # Never sampled
        return time.time() - self.last_sampled


class ReplayBuffer:
    """Prioritized Replay Buffer with importance-based sampling.
    
    Features:
    - Prioritized sampling based on feedback, importance, and age
    - Efficient storage with LRU-like cache eviction
    - Support for batch sampling and importance-weighted replay
    """
    
    def __init__(self, config):
        """Initialize the replay buffer.
        
        Args:
            config: Global config object
        """
        self.config = config
        self.storage_manager = None  # Will be set during initialize()
        
        # Buffer parameters
        self.max_size = config.training.general.max_queue_size * 10  # Default to 10x the queue size
        self.alpha = 0.6  # Priority exponent (0 = uniform, 1 = fully prioritized)
        self.beta = 0.4  # Initial importance sampling correction (0 = no correction, 1 = full correction)
        self.beta_increment = 0.001  # How much to increase beta during optimization
        
        # Buffer storage (dict for O(1) lookups by ID)
        self.buffer = {}  # id -> ReplayItem
        self.priorities = []  # Priority queue for sampling
        
        # Buffer statistics
        self.stats = {
            "total_added": 0,
            "total_sampled": 0,
            "total_optimized": 0,
            "avg_priority": 0.0,
            "size": 0,
            "evictions": 0
        }
        
        # Synchronization
        self.lock = asyncio.Lock()
        self.running = False
    
    async def initialize(self, storage_manager=None) -> bool:
        """Initialize the replay buffer.
        
        Args:
            storage_manager: Optional storage manager for persisting buffer
            
        Returns:
            bool: True if initialization was successful
        """
        try:
            logger.info("Initializing replay buffer")
            self.running = True
            
            if storage_manager:
                self.storage_manager = storage_manager
                
                # Try to load persisted buffer
                try:
                    persisted_data = await self.storage_manager.load_data(
                        "replay_buffer", 
                        "latest"
                    )
                    if persisted_data and "items" in persisted_data:
                        loaded_count = 0
                        for item_data in persisted_data["items"]:
                            try:
                                item = ReplayItem(**item_data)
                                self.buffer[item.metadata.get("id", f"item_{len(self.buffer)}")] = item
                                heapq.heappush(self.priorities, (-item.priority, item.metadata.get("id")))
                                loaded_count += 1
                            except Exception as item_e:
                                logger.warning(f"Error loading item: {item_e}")
                                continue
                        
                        logger.info(f"Loaded {loaded_count} items from persisted replay buffer")
                except Exception as e:
                    logger.warning(f"Could not load persisted buffer: {e}")
            
            logger.info("Replay buffer initialized", size=len(self.buffer))
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize replay buffer: {e}")
            return False
    
    async def add_sample(self, prompt: str, response: str, feedback_score: float,
                        importance_weight: float = 1.0, metadata: Dict[str, Any] = None) -> bool:
        """Add a sample to the replay buffer.
        
        Args:
            prompt: The input prompt
            response: The model's response
            feedback_score: Feedback score (higher = better)
            importance_weight: Sample importance for prioritization
            metadata: Additional metadata for the sample
            
        Returns:
            bool: True if sample was added successfully
        """
        try:
            if not self.running:
                logger.warning("Replay buffer not running")
                return False
                
            async with self.lock:
                # Generate a unique ID if not provided
                metadata = metadata or {}
                item_id = metadata.get("id") or metadata.get("request_id") or f"item_{int(time.time())}_{len(self.buffer)}"
                
                # Calculate priority
                # Priority is a function of feedback, importance, and recency
                abs_feedback = abs(feedback_score)  # Both very positive and very negative are interesting
                priority = abs_feedback * importance_weight
                
                # Create replay item
                item = ReplayItem(
                    prompt=prompt,
                    response=response,
                    feedback_score=feedback_score,
                    importance_weight=importance_weight,
                    priority=priority,
                    timestamp=time.time(),
                    last_sampled=None,
                    sampling_count=0,
                    metadata={**metadata, "id": item_id}
                )
                
                # Add to buffer
                self.buffer[item_id] = item
                heapq.heappush(self.priorities, (-priority, item_id))
                
                # Update stats
                self.stats["total_added"] += 1
                self.stats["size"] = len(self.buffer)
                self.stats["avg_priority"] = sum(-p for p, _ in self.priorities) / len(self.priorities) if self.priorities else 0
                
                # Evict if over max size
                await self._maybe_evict()
                
                logger.debug(f"Added sample to replay buffer", id=item_id, priority=priority)
                return True
                
        except Exception as e:
            logger.error(f"Error adding sample to replay buffer: {e}")
            return False
    
    async def get_batch(self, batch_size: int = 10) -> List[Dict[str, Any]]:
        """Get a batch of samples from the replay buffer using prioritized sampling.
        
        Args:
            batch_size: Number of samples to retrieve
            
        Returns:
            List of dictionaries containing sample data
        """
        try:
            if not self.running:
                logger.warning("Replay buffer not running")
                return []
                
            async with self.lock:
                if not self.buffer:
                    return []
                
                # Limit batch size to buffer size
                batch_size = min(batch_size, len(self.buffer))
                
                # Extract top N priorities
                top_items = []
                top_ids = set()
                
                # First pass: get top 2x batch_size candidates (for diversity)
                candidates = []
                candidate_count = min(batch_size * 2, len(self.priorities))
                
                for _ in range(candidate_count):
                    if not self.priorities:
                        break
                    priority, item_id = heapq.heappop(self.priorities)
                    if item_id in self.buffer:  # Ensure item still exists
                        candidates.append((priority, item_id))
                
                # Second pass: probabilistically sample from candidates
                if candidates:
                    # Convert priorities to probabilities
                    priorities = np.array([-p for p, _ in candidates]) ** self.alpha
                    probs = priorities / priorities.sum()
                    
                    # Sample without replacement
                    indices = np.random.choice(
                        len(candidates), 
                        size=min(batch_size, len(candidates)), 
                        replace=False, 
                        p=probs
                    )
                    
                    selected_candidates = [candidates[i] for i in indices]
                    top_ids = {item_id for _, item_id in selected_candidates}
                    
                    # Put unselected candidates back
                    for priority, item_id in candidates:
                        if item_id not in top_ids:
                            heapq.heappush(self.priorities, (priority, item_id))
                
                # Get the actual items and update their metadata
                now = time.time()
                batch = []
                
                for item_id in top_ids:
                    item = self.buffer[item_id]
                    item.last_sampled = now
                    item.sampling_count += 1
                    
                    # Update priority based on sampling history
                    # Items sampled too often get reduced priority
                    sampling_decay = 0.95 ** min(item.sampling_count, 10)
                    adjusted_priority = item.priority * sampling_decay
                    
                    # Re-add to priority queue with updated priority
                    heapq.heappush(self.priorities, (-adjusted_priority, item_id))
                    
                    # Add to batch in dictionary form
                    batch.append({
                        "prompt": item.prompt,
                        "response": item.response,
                        "feedback_score": item.feedback_score,
                        "importance_weight": item.importance_weight,
                        "metadata": item.metadata
                    })
                
                # Update stats
                self.stats["total_sampled"] += len(batch)
                
                logger.debug(f"Retrieved batch from replay buffer", batch_size=len(batch))
                return batch
                
        except Exception as e:
            logger.error(f"Error getting batch from replay buffer: {e}")
            return []
    
    async def optimize(self) -> bool:
        """Optimize the replay buffer.
        
        Performs:
        - Priority recalculation
        - Removal of stale items
        - Beta annealing for importance sampling
        - Persistence (if storage manager available)
        
        Returns:
            bool: True if optimization was successful
        """
        try:
            if not self.running:
                logger.warning("Replay buffer not running")
                return False
                
            async with self.lock:
                logger.info("Optimizing replay buffer", size=len(self.buffer))
                
                # Recalculate all priorities based on age and sampling history
                now = time.time()
                new_priorities = []
                
                for item_id, item in list(self.buffer.items()):
                    # Items get less interesting as they age (unless they're rare)
                    age_factor = 0.98 ** (item.age / (60 * 60 * 24))  # Decay by 2% per day
                    
                    # Items sampled very frequently become less interesting
                    sampling_factor = 0.9 ** min(item.sampling_count, 20)  # Max 20x decay
                    
                    # Items never sampled get a boost
                    novelty_factor = 1.2 if item.sampling_count == 0 else 1.0
                    
                    # Recalculate priority
                    recalculated_priority = item.priority * age_factor * sampling_factor * novelty_factor
                    
                    # Add to new priority queue
                    new_priorities.append((-recalculated_priority, item_id))
                
                # Replace priority queue
                self.priorities = []
                for priority_item in new_priorities:
                    heapq.heappush(self.priorities, priority_item)
                
                # Increase beta for importance sampling correction
                self.beta = min(1.0, self.beta + self.beta_increment)
                
                # Update stats
                self.stats["total_optimized"] += 1
                self.stats["avg_priority"] = sum(-p for p, _ in self.priorities) / len(self.priorities) if self.priorities else 0
                
                # Persist if storage manager available
                if self.storage_manager:
                    # Only store up to 1000 items to avoid excessive storage
                    items_to_store = sorted(
                        list(self.buffer.values()),
                        key=lambda x: x.priority,
                        reverse=True
                    )[:1000]
                    
                    await self.storage_manager.save_data(
                        "replay_buffer",
                        "latest",
                        {
                            "items": [vars(item) for item in items_to_store],
                            "stats": self.stats,
                            "timestamp": now
                        }
                    )
                
                logger.info("Replay buffer optimization complete", size=len(self.buffer))
                return True
                
        except Exception as e:
            logger.error(f"Error optimizing replay buffer: {e}")
            return False
    
    async def get_size(self) -> int:
        """Get the current size of the replay buffer.
        
        Returns:
            int: Number of items in the buffer
        """
        return len(self.buffer)
    
    async def get_status(self) -> Dict[str, Any]:
        """Get status information about the replay buffer.
        
        Returns:
            Dict with status information
        """
        return {
            "running": self.running,
            "size": len(self.buffer),
            "max_size": self.max_size,
            "alpha": self.alpha,
            "beta": self.beta,
            "stats": self.stats
        }
    
    async def get_checkpoint(self) -> Dict[str, Any]:
        """Get checkpoint data for the replay buffer.
        
        Returns:
            Dict with checkpoint data
        """
        async with self.lock:
            # Only checkpoint up to 1000 highest priority items
            items_to_checkpoint = sorted(
                list(self.buffer.values()),
                key=lambda x: x.priority,
                reverse=True
            )[:1000]
            
            return {
                "items": [vars(item) for item in items_to_checkpoint],
                "alpha": self.alpha,
                "beta": self.beta,
                "stats": self.stats
            }
    
    async def load_checkpoint(self, checkpoint_data: Dict[str, Any]) -> bool:
        """Load checkpoint data into the replay buffer.
        
        Args:
            checkpoint_data: Checkpoint data to load
            
        Returns:
            bool: True if checkpoint was loaded successfully
        """
        try:
            async with self.lock:
                # Clear existing buffer
                self.buffer = {}
                self.priorities = []
                
                # Load parameters
                if "alpha" in checkpoint_data:
                    self.alpha = checkpoint_data["alpha"]
                if "beta" in checkpoint_data:
                    self.beta = checkpoint_data["beta"]
                if "stats" in checkpoint_data:
                    self.stats = checkpoint_data["stats"]
                
                # Load items
                if "items" in checkpoint_data:
                    loaded_count = 0
                    for item_data in checkpoint_data["items"]:
                        item = ReplayItem(**item_data)
                        item_id = item.metadata.get("id", f"item_{len(self.buffer)}")
                        self.buffer[item_id] = item
                        heapq.heappush(self.priorities, (-item.priority, item_id))
                        loaded_count += 1
                    
                    logger.info(f"Loaded {loaded_count} items from checkpoint")
                
                self.stats["size"] = len(self.buffer)
                return True
                
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear the replay buffer.
        
        Returns:
            bool: True if the buffer was cleared successfully
        """
        try:
            async with self.lock:
                self.buffer = {}
                self.priorities = []
                self.stats["size"] = 0
                self.stats["evictions"] = 0
                logger.info("Replay buffer cleared")
                return True
                
        except Exception as e:
            logger.error(f"Error clearing replay buffer: {e}")
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown the replay buffer.
        
        Returns:
            bool: True if shutdown was successful
        """
        try:
            self.running = False
            
            # Persist buffer if storage manager available
            if self.storage_manager:
                try:
                    await self.optimize()  # This will persist the buffer
                except Exception as e:
                    logger.warning(f"Error persisting buffer during shutdown: {e}")
            
            logger.info("Replay buffer shutdown complete")
            return True
            
        except Exception as e:
            logger.error(f"Error shutting down replay buffer: {e}")
            return False
    
    async def _maybe_evict(self) -> None:
        """Evict items if buffer is over max size."""
        # Check if we need to evict
        if len(self.buffer) <= self.max_size:
            return
        
        # Number of items to evict
        to_evict = len(self.buffer) - self.max_size
        
        # Criteria for eviction:
        # 1. Low priority items
        # 2. Old items that have been sampled many times
        
        # Get all items sorted by:
        # - First by sampling_count / age ratio (higher = more stale)
        # - Then by priority (lower = less valuable)
        items_by_staleness = sorted(
            list(self.buffer.items()),
            key=lambda x: (
                x[1].sampling_count / max(x[1].age, 1),  # Staleness
                -x[1].priority  # Negative priority (we want low priority items first)
            ),
            reverse=True
        )
        
        # Evict the stalest, lowest priority items
        for i in range(min(to_evict, len(items_by_staleness))):
            item_id, _ = items_by_staleness[i]
            del self.buffer[item_id]
            self.stats["evictions"] += 1
        
        # Rebuild priority queue (inefficient but simple)
        # In a production system, you'd use a more efficient approach
        self.priorities = []
        for item_id, item in self.buffer.items():
            heapq.heappush(self.priorities, (-item.priority, item_id))
        
        logger.debug(f"Evicted {to_evict} items from replay buffer", 
                   new_size=len(self.buffer)) 