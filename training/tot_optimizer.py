"""Tree-of-Thought Optimizer for structured reasoning and response improvement."""

import asyncio
import time
import math
import re
import heapq
from collections import deque
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import threading

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import structlog

# Import BERTScore for evaluation
from bert_score import BERTScorer

logger = structlog.get_logger(__name__)


class SearchStrategy(Enum):
    """Search strategy for Tree-of-Thought exploration."""
    BFS = "bfs"  # Breadth-first search
    DFS = "dfs"  # Depth-first search
    BEAM = "beam"  # Beam search
    MCTS = "mcts"  # Monte Carlo Tree Search (future)


@dataclass
class ThoughtNode:
    """Represents a node in the thought tree."""
    node_id: str
    parent_id: Optional[str]
    depth: int
    thought: str
    response: str
    score: float
    metrics: Dict[str, float]
    children: List[str] = field(default_factory=list)
    visits: int = 0
    is_terminal: bool = False
    pruned: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationResult:
    """Result of Tree-of-Thought optimization."""
    original_response: str
    optimized_response: str
    improvement_score: float
    exploration_path: List[str]
    total_nodes_explored: int
    optimization_time: float
    best_node: ThoughtNode
    all_nodes: Dict[str, ThoughtNode]


@dataclass
class ToTMetrics:
    """Metrics for Tree-of-Thought optimization."""
    total_optimizations: int = 0
    successful_optimizations: int = 0
    failed_optimizations: int = 0
    avg_improvement_score: float = 0.0
    avg_nodes_explored: float = 0.0
    avg_optimization_time: float = 0.0
    total_nodes_created: int = 0
    total_nodes_pruned: int = 0


class ToTOptimizer:
    """Tree-of-Thought optimizer for improving model responses."""
    
    def __init__(self, model: nn.Module, tokenizer: AutoTokenizer, config):
        # Generation model
        self.model = model
        self.tokenizer = tokenizer
        self.config = config  # Global config
        self.tot_config = config.training.tot if hasattr(config.training, 'tot') else None
        
        # Default configuration if not provided
        if self.tot_config is None:
            self.tot_config = self._get_default_config()
        
        # Optimization parameters
        self.max_depth = getattr(self.tot_config, 'max_depth', 3)
        self.branching_factor = getattr(self.tot_config, 'branching_factor', 3)
        self.beam_width = getattr(self.tot_config, 'beam_width', 5)
        self.evaluation_samples = getattr(self.tot_config, 'evaluation_samples', 5)
        self.improvement_threshold = getattr(self.tot_config, 'improvement_threshold', 0.1)
        self.max_optimization_time = getattr(self.tot_config, 'max_optimization_time_seconds', 120)
        self.search_strategy = SearchStrategy(getattr(self.tot_config, 'search_strategy', 'bfs'))
        self.pruning_enabled = getattr(self.tot_config, 'pruning_enabled', True)
        self.pruning_threshold = getattr(self.tot_config, 'pruning_threshold', 0.05)
        
        # Evaluation metrics configuration
        self.evaluation_metrics = getattr(self.tot_config, 'evaluation_metrics', 
                                        ['coherence', 'relevance', 'quality', 'bert_score'])
        self.metric_weights = getattr(self.tot_config, 'metric_weights', 
                                    [0.3, 0.3, 0.2, 0.2])
        
        # Normalize weights
        weight_sum = sum(self.metric_weights)
        self.metric_weights = [w / weight_sum for w in self.metric_weights]
        
        # State
        self.running = False
        self.metrics = ToTMetrics()
        self.optimization_history: List[OptimizationResult] = []
        self.device = next(model.parameters()).device
        
        # Separate evaluation model (same architecture but separate instance)
        # This prevents the model from gaming its own rewards
        self.eval_model = None  # Will be initialized in initialize()
        
        # BERTScore for semantic evaluation
        self.bert_scorer = None  # Will be initialized in initialize()
        
        # Thought generation prompts
        self.thought_prompts = self._initialize_thought_prompts()
        
    def _get_default_config(self):
        """Get default configuration for ToT optimizer."""
        class DefaultConfig:
            max_depth = 3
            branching_factor = 3
            beam_width = 5
            evaluation_samples = 5
            improvement_threshold = 0.1
            max_optimization_time_seconds = 120
            search_strategy = 'bfs'
            pruning_enabled = True
            pruning_threshold = 0.05
            evaluation_metrics = ['coherence', 'relevance', 'quality', 'bert_score']
            metric_weights = [0.3, 0.3, 0.2, 0.2]
        
        return DefaultConfig()
    
    def _initialize_thought_prompts(self) -> List[str]:
        """Initialize prompts for generating thoughts at each step."""
        return [
            "Let me think about this differently: ",
            "Another approach would be: ",
            "Considering the context more carefully: ",
            "Breaking this down further: ",
            "From a different perspective: ",
            "To improve clarity: ",
            "To be more specific: ",
            "Addressing potential issues: ",
            "Enhancing the explanation: ",
            "Refining the response: "
        ]
    
    async def initialize(self) -> bool:
        """Initialize the ToT optimizer."""
        try:
            logger.info("Initializing Tree-of-Thought optimizer",
                       max_depth=self.max_depth,
                       branching_factor=self.branching_factor,
                       search_strategy=self.search_strategy.value)
            
            # Initialize separate evaluation model
            model_name = self.config.model.name
            trust_remote_code = getattr(self.config.model, 'trust_remote_code', True)
            
            logger.info("Loading separate evaluation model to prevent reward gaming")
            self.eval_model = type(self.model).from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
                device_map="auto",  # Use the same device mapping strategy
                torch_dtype=torch.bfloat16  # Use same precision
            )
            
            # Freeze evaluation model to prevent training
            for param in self.eval_model.parameters():
                param.requires_grad = False
            
            # Initialize BERTScore
            logger.info("Initializing BERTScore for semantic evaluation")
            self.bert_scorer = BERTScorer(
                lang="en", 
                rescale_with_baseline=True,
                device=self.device
            )
            
            self.running = True
            
            logger.info("ToT optimizer initialized successfully with separate evaluation model")
            return True
            
        except Exception as e:
            logger.error("Failed to initialize ToT optimizer", error=str(e))
            return False
    
    async def optimize_response(
        self,
        prompt: str,
        bad_response: str,
        feedback_score: float,
        context: Optional[str] = None
    ) -> Optional[str]:
        """Optimize a response using Tree-of-Thought search."""
        try:
            start_time = time.time()
            
            # Create optimization context
            optimization_context = self._create_optimization_context(
                prompt, bad_response, feedback_score, context
            )
            
            # Run Tree-of-Thought optimization
            result = await self._run_tot_optimization(optimization_context)
            
            if result and result.improvement_score > self.improvement_threshold:
                # Update metrics
                self._update_metrics(result)
                
                # Store in history
                self.optimization_history.append(result)
                if len(self.optimization_history) > 100:
                    self.optimization_history.pop(0)
                
                logger.info("Response optimized successfully",
                           improvement_score=result.improvement_score,
                           nodes_explored=result.total_nodes_explored,
                           optimization_time=result.optimization_time)
                
                return result.optimized_response
            else:
                logger.debug("No significant improvement found",
                            improvement_score=result.improvement_score if result else 0)
                return None
                
        except Exception as e:
            logger.error("Error optimizing response", error=str(e))
            self.metrics.failed_optimizations += 1
            return None
    
    def _create_optimization_context(
        self,
        prompt: str,
        bad_response: str,
        feedback_score: float,
        context: Optional[str]
    ) -> Dict[str, Any]:
        """Create context for optimization."""
        return {
            'prompt': prompt,
            'bad_response': bad_response,
            'feedback_score': feedback_score,
            'context': context or "",
            'optimization_goal': f"Improve this response (current score: {feedback_score:.2f})",
            'timestamp': time.time()
        }
    
    async def _run_tot_optimization(
        self,
        context: Dict[str, Any]
    ) -> Optional[OptimizationResult]:
        """Run Tree-of-Thought optimization process."""
        try:
            # Initialize root node
            root_node = ThoughtNode(
                node_id="root",
                parent_id=None,
                depth=0,
                thought="Initial response",
                response=context['bad_response'],
                score=context['feedback_score'],
                metrics=await self._evaluate_response(
                    context['prompt'], 
                    context['bad_response']
                )
            )
            
            # Initialize search structures
            all_nodes = {"root": root_node}
            best_node = root_node
            nodes_explored = 1
            
            # Choose search strategy
            if self.search_strategy == SearchStrategy.BFS:
                best_node = await self._bfs_search(
                    root_node, context, all_nodes
                )
            elif self.search_strategy == SearchStrategy.DFS:
                best_node = await self._dfs_search(
                    root_node, context, all_nodes, self.max_depth
                )
            elif self.search_strategy == SearchStrategy.BEAM:
                best_node = await self._beam_search(
                    root_node, context, all_nodes
                )
            else:
                logger.warning(f"Unknown search strategy: {self.search_strategy}")
                return None
            
            # Calculate improvement
            improvement_score = best_node.score - root_node.score
            
            # Build exploration path
            exploration_path = self._build_path(best_node, all_nodes)
            
            return OptimizationResult(
                original_response=context['bad_response'],
                optimized_response=best_node.response,
                improvement_score=improvement_score,
                exploration_path=exploration_path,
                total_nodes_explored=len(all_nodes),
                optimization_time=time.time() - context['timestamp'],
                best_node=best_node,
                all_nodes=all_nodes
            )
            
        except Exception as e:
            logger.error("Error in ToT optimization", error=str(e))
            return None
    
    async def _bfs_search(
        self,
        root: ThoughtNode,
        context: Dict[str, Any],
        all_nodes: Dict[str, ThoughtNode]
    ) -> ThoughtNode:
        """Breadth-first search through thought tree."""
        queue = deque([root])
        best_node = root
        start_time = time.time()
        
        while queue and (time.time() - start_time) < self.max_optimization_time:
            if len(all_nodes) >= 1000:  # Safety limit
                logger.warning("Node limit reached in BFS")
                break
                
            current = queue.popleft()
            
            # Skip if pruned or at max depth
            if current.pruned or current.depth >= self.max_depth:
                continue
            
            # Generate children
            children = await self._generate_children(current, context, all_nodes)
            
            for child in children:
                if child.score > best_node.score:
                    best_node = child
                
                # Prune if enabled
                if self.pruning_enabled and self._should_prune(child, best_node):
                    child.pruned = True
                else:
                    queue.append(child)
        
        return best_node
    
    async def _dfs_search(
        self,
        node: ThoughtNode,
        context: Dict[str, Any],
        all_nodes: Dict[str, ThoughtNode],
        max_depth: int
    ) -> ThoughtNode:
        """Depth-first search through thought tree."""
        if node.depth >= max_depth or node.pruned:
            return node
        
        # Generate children
        children = await self._generate_children(node, context, all_nodes)
        
        best_node = node
        for child in children:
            if child.score > best_node.score:
                best_node = child
            
            # Recursive exploration
            if not self._should_prune(child, best_node):
                child_best = await self._dfs_search(child, context, all_nodes, max_depth)
                if child_best.score > best_node.score:
                    best_node = child_best
        
        return best_node
    
    async def _beam_search(
        self,
        root: ThoughtNode,
        context: Dict[str, Any],
        all_nodes: Dict[str, ThoughtNode]
    ) -> ThoughtNode:
        """Beam search through thought tree."""
        beam = [(-root.score, root)]  # Min heap (negative score for max)
        best_node = root
        
        for depth in range(self.max_depth):
            next_beam = []
            
            # Expand current beam
            for _, node in beam:
                if node.pruned:
                    continue
                    
                children = await self._generate_children(node, context, all_nodes)
                
                for child in children:
                    if child.score > best_node.score:
                        best_node = child
                    
                    if not self._should_prune(child, best_node):
                        heapq.heappush(next_beam, (-child.score, child))
            
            # Keep top beam_width nodes
            beam = heapq.nsmallest(self.beam_width, next_beam)
            
            if not beam:
                break
        
        return best_node
    
    async def _generate_children(
        self,
        parent: ThoughtNode,
        context: Dict[str, Any],
        all_nodes: Dict[str, ThoughtNode]
    ) -> List[ThoughtNode]:
        """Generate child nodes for a parent node."""
        children = []
        
        for i in range(min(self.branching_factor, len(self.thought_prompts))):
            try:
                # Generate thought
                thought_prompt = self.thought_prompts[i % len(self.thought_prompts)]
                thought = await self._generate_thought(
                    parent, context, thought_prompt
                )
                
                # Generate improved response based on thought
                improved_response = await self._generate_improved_response(
                    context['prompt'], parent.response, thought
                )
                
                # Evaluate the new response
                metrics = await self._evaluate_response(
                    context['prompt'], improved_response
                )
                score = self._calculate_score(metrics)
                
                # Create child node
                child = ThoughtNode(
                    node_id=f"{parent.node_id}_child_{i}",
                    parent_id=parent.node_id,
                    depth=parent.depth + 1,
                    thought=thought,
                    response=improved_response,
                    score=score,
                    metrics=metrics
                )
                
                parent.children.append(child.node_id)
                all_nodes[child.node_id] = child
                children.append(child)
                
                self.metrics.total_nodes_created += 1
                
            except Exception as e:
                logger.error("Error generating child node", error=str(e))
                continue
        
        return children
    
    async def _generate_thought(
        self,
        parent: ThoughtNode,
        context: Dict[str, Any],
        thought_prompt: str
    ) -> str:
        """Generate a thought for improving the response."""
        try:
            # Construct prompt for thought generation
            prompt = f"""Given this question: {context['prompt']}

Current response: {parent.response}

The response received a feedback score of {context['feedback_score']:.2f}.

{thought_prompt}"""

            # Generate thought using the model
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.8,
                    do_sample=True,
                    top_p=0.9
                )
            
            thought = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            thought = thought[len(prompt):].strip()
            
            return thought
            
        except Exception as e:
            logger.error("Error generating thought", error=str(e))
            return "Consider alternative approaches."
    
    async def _generate_improved_response(
        self,
        original_prompt: str,
        current_response: str,
        thought: str
    ) -> str:
        """Generate an improved response based on the thought."""
        try:
            # Construct prompt for response improvement
            prompt = f"""Question: {original_prompt}

Current response: {current_response}

Improvement strategy: {thought}

Improved response:"""

            # Generate improved response
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=300,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9
                )
            
            improved = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            improved = improved[len(prompt):].strip()
            
            return improved
            
        except Exception as e:
            logger.error("Error generating improved response", error=str(e))
            return current_response
    
    async def _evaluate_response(
        self,
        prompt: str,
        response: str
    ) -> Dict[str, float]:
        """Evaluate a response on multiple metrics."""
        metrics = {}
        
        for metric in self.evaluation_metrics:
            if metric == 'coherence':
                score = await self._evaluate_coherence(prompt, response)
            elif metric == 'relevance':
                score = await self._evaluate_relevance(prompt, response)
            elif metric == 'quality':
                score = await self._evaluate_quality(response)
            elif metric == 'bert_score':
                score = await self._evaluate_bert_score(prompt, response)
            else:
                score = 0.5  # Default neutral score
            
            metrics[metric] = score
        
        return metrics
    
    async def _evaluate_coherence(self, prompt: str, response: str) -> float:
        """Evaluate how coherent a response is with respect to the prompt."""
        try:
            # Prepare evaluation prompt
            eval_prompt = f"""
            Evaluate the coherence of this response to the given prompt.
            Prompt: {prompt}
            
            Response: {response}
            
            Rate the coherence from 0.0 to 1.0, where:
            - 0.0: Completely incoherent, no logical flow
            - 0.5: Somewhat coherent, but with issues
            - 1.0: Perfectly coherent, clear logical flow
            
            Coherence score:
            """
            
            # Generate coherence evaluation using the separate evaluation model
            inputs = self.tokenizer(eval_prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                output = self.eval_model.generate(
                    **inputs,
                    max_new_tokens=10,
                    temperature=0.1
                )
            
            # Extract score
            result = self.tokenizer.decode(output[0][inputs['input_ids'].shape[1]:])
            
            # Parse score from result
            score_match = re.search(r"(\d+\.?\d*)", result)
            if score_match:
                score = float(score_match.group(1))
                return min(max(score, 0.0), 1.0)  # Clamp to [0.0, 1.0]
            
            return 0.5  # Default if parsing fails
        except Exception as e:
            logger.error("Error evaluating coherence", error=str(e))
            return 0.5  # Default on error
    
    async def _evaluate_relevance(self, prompt: str, response: str) -> float:
        """Evaluate how relevant a response is to the prompt."""
        try:
            # Prepare evaluation prompt
            eval_prompt = f"""
            Evaluate how relevant this response is to the given prompt.
            Prompt: {prompt}
            
            Response: {response}
            
            Rate the relevance from 0.0 to 1.0, where:
            - 0.0: Completely irrelevant, doesn't address the prompt
            - 0.5: Somewhat relevant, but misses key points
            - 1.0: Highly relevant, directly addresses all aspects of the prompt
            
            Relevance score:
            """
            
            # Generate relevance evaluation using the separate evaluation model
            inputs = self.tokenizer(eval_prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                output = self.eval_model.generate(
                    **inputs,
                    max_new_tokens=10,
                    temperature=0.1
                )
            
            # Extract score
            result = self.tokenizer.decode(output[0][inputs['input_ids'].shape[1]:])
            
            # Parse score from result
            score_match = re.search(r"(\d+\.?\d*)", result)
            if score_match:
                score = float(score_match.group(1))
                return min(max(score, 0.0), 1.0)  # Clamp to [0.0, 1.0]
            
            return 0.5  # Default if parsing fails
        except Exception as e:
            logger.error("Error evaluating relevance", error=str(e))
            return 0.5  # Default on error
    
    async def _evaluate_quality(self, response: str) -> float:
        """Evaluate overall response quality."""
        try:
            # Prepare evaluation prompt
            eval_prompt = f"""
            Evaluate the overall quality of this text:
            
            Text: {response}
            
            Rate the quality from 0.0 to 1.0, where:
            - 0.0: Poor quality (errors, unclear, poorly structured)
            - 0.5: Average quality
            - 1.0: Excellent quality (clear, informative, well-structured)
            
            Quality score:
            """
            
            # Generate quality evaluation using the separate evaluation model
            inputs = self.tokenizer(eval_prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                output = self.eval_model.generate(
                    **inputs,
                    max_new_tokens=10,
                    temperature=0.1
                )
            
            # Extract score
            result = self.tokenizer.decode(output[0][inputs['input_ids'].shape[1]:])
            
            # Parse score from result
            score_match = re.search(r"(\d+\.?\d*)", result)
            if score_match:
                score = float(score_match.group(1))
                return min(max(score, 0.0), 1.0)  # Clamp to [0.0, 1.0]
            
            return 0.5  # Default if parsing fails
        except Exception as e:
            logger.error("Error evaluating quality", error=str(e))
            return 0.5  # Default on error
    
    async def _evaluate_bert_score(self, prompt: str, response: str) -> float:
        """Evaluate semantic similarity using BERTScore."""
        try:
            if self.bert_scorer is None:
                logger.warning("BERTScore not initialized, skipping evaluation")
                return 0.5  # Default if BERTScore not available
            
            # Calculate BERTScore (F1) between prompt and response
            with torch.no_grad():
                P, R, F1 = self.bert_scorer.score([response], [prompt])
                
                # We use F1 as the primary score
                bert_score = F1.mean().item()
                
                # Normalize to [0, 1] range if needed
                # BERTScore is already normalized with rescale_with_baseline=True
                
                return bert_score
                
        except Exception as e:
            logger.error("Error calculating BERTScore", error=str(e))
            return 0.5  # Default on error
    
    def _calculate_score(self, metrics: Dict[str, float]) -> float:
        """Calculate weighted score from metrics."""
        score = 0.0
        for i, metric in enumerate(self.evaluation_metrics):
            if metric in metrics and i < len(self.metric_weights):
                score += metrics[metric] * self.metric_weights[i]
        return score
    
    def _should_prune(self, node: ThoughtNode, best_node: ThoughtNode) -> bool:
        """Determine if a node should be pruned."""
        if not self.pruning_enabled:
            return False
        
        # Prune if score is significantly worse than best
        if node.score < best_node.score - self.pruning_threshold:
            self.metrics.total_nodes_pruned += 1
            return True
        
        # Prune if metrics are all below threshold
        if all(score < 0.3 for score in node.metrics.values()):
            self.metrics.total_nodes_pruned += 1
            return True
        
        return False
    
    def _build_path(
        self,
        node: ThoughtNode,
        all_nodes: Dict[str, ThoughtNode]
    ) -> List[str]:
        """Build path from root to node."""
        path = []
        current = node
        
        while current:
            path.append(current.node_id)
            if current.parent_id and current.parent_id in all_nodes:
                current = all_nodes[current.parent_id]
            else:
                break
        
        return list(reversed(path))
    
    def _update_metrics(self, result: OptimizationResult):
        """Update metrics after optimization."""
        self.metrics.total_optimizations += 1
        
        if result.improvement_score > 0:
            self.metrics.successful_optimizations += 1
        
        # Update averages
        n = self.metrics.total_optimizations
        self.metrics.avg_improvement_score = (
            (self.metrics.avg_improvement_score * (n - 1) + result.improvement_score) / n
        )
        self.metrics.avg_nodes_explored = (
            (self.metrics.avg_nodes_explored * (n - 1) + result.total_nodes_explored) / n
        )
        self.metrics.avg_optimization_time = (
            (self.metrics.avg_optimization_time * (n - 1) + result.optimization_time) / n
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get current ToT optimizer status."""
        return {
            "running": self.running,
            "search_strategy": self.search_strategy.value,
            "max_depth": self.max_depth,
            "branching_factor": self.branching_factor,
            "metrics": {
                "total_optimizations": self.metrics.total_optimizations,
                "successful_optimizations": self.metrics.successful_optimizations,
                "failed_optimizations": self.metrics.failed_optimizations,
                "success_rate": (
                    self.metrics.successful_optimizations / max(1, self.metrics.total_optimizations)
                ),
                "avg_improvement_score": self.metrics.avg_improvement_score,
                "avg_nodes_explored": self.metrics.avg_nodes_explored,
                "avg_optimization_time": self.metrics.avg_optimization_time,
                "total_nodes_created": self.metrics.total_nodes_created,
                "total_nodes_pruned": self.metrics.total_nodes_pruned
            },
            "recent_optimizations": len(self.optimization_history),
            "evaluation_metrics": self.evaluation_metrics,
            "metric_weights": self.metric_weights
        }
    
    async def get_checkpoint(self) -> Dict[str, Any]:
        """Get checkpoint data for saving."""
        return {
            "metrics": {
                "total_optimizations": self.metrics.total_optimizations,
                "successful_optimizations": self.metrics.successful_optimizations,
                "failed_optimizations": self.metrics.failed_optimizations,
                "avg_improvement_score": self.metrics.avg_improvement_score,
                "avg_nodes_explored": self.metrics.avg_nodes_explored,
                "avg_optimization_time": self.metrics.avg_optimization_time,
                "total_nodes_created": self.metrics.total_nodes_created,
                "total_nodes_pruned": self.metrics.total_nodes_pruned
            },
            "search_strategy": self.search_strategy.value,
            "optimization_history": [
                {
                    "original_response": r.original_response[:100],  # Truncate for storage
                    "optimized_response": r.optimized_response[:100],
                    "improvement_score": r.improvement_score,
                    "nodes_explored": r.total_nodes_explored,
                    "optimization_time": r.optimization_time
                }
                for r in self.optimization_history[-10:]  # Keep last 10
            ]
        }
    
    async def load_checkpoint(self, checkpoint: Dict[str, Any]) -> bool:
        """Load checkpoint data."""
        try:
            if "metrics" in checkpoint:
                metrics_data = checkpoint["metrics"]
                self.metrics.total_optimizations = metrics_data.get("total_optimizations", 0)
                self.metrics.successful_optimizations = metrics_data.get("successful_optimizations", 0)
                self.metrics.failed_optimizations = metrics_data.get("failed_optimizations", 0)
                self.metrics.avg_improvement_score = metrics_data.get("avg_improvement_score", 0.0)
                self.metrics.avg_nodes_explored = metrics_data.get("avg_nodes_explored", 0.0)
                self.metrics.avg_optimization_time = metrics_data.get("avg_optimization_time", 0.0)
                self.metrics.total_nodes_created = metrics_data.get("total_nodes_created", 0)
                self.metrics.total_nodes_pruned = metrics_data.get("total_nodes_pruned", 0)
            
            logger.info("ToT checkpoint loaded successfully")
            return True
            
        except Exception as e:
            logger.error("Error loading ToT checkpoint", error=str(e))
            return False
    
    async def shutdown(self):
        """Shutdown the ToT optimizer."""
        logger.info("Shutting down ToT optimizer")
        self.running = False
        self.optimization_history.clear()
        logger.info("ToT optimizer shutdown complete") 