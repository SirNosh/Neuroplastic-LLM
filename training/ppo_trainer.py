"""Proximal-Policy Optimisation (PPO) trainer for RLHF fine-tuning.

This module fine-tunes a LoRA-enabled policy model from user feedback
traces stored in Kafka or another feedback store.  The trainer is
scheduled (e.g. nightly) and therefore runs independently of
``OnlineTrainer``.  It only touches LoRA adapters – the base weights
stay frozen – ensuring safe hot-reload into the serving cluster.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedModel, PreTrainedTokenizer)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from trl import PPOConfig as TRLPPOConfig, PPOTrainer
import structlog

logger = structlog.get_logger(__name__)


class PPOTrainerWrapper:
    """Wrapper around TRL PPOTrainer adding LoRA checkpoint utilities."""

    def __init__(
        self,
        base_model_name: str,
        tokenizer: PreTrainedTokenizer,
        ppo_cfg: TRLPPOConfig,
        lora_cfg: LoraConfig,
        device: str | torch.device = "cuda"
    ):
        # Create policy + ref model with LoRA adapters (trainable)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
        policy_model = get_peft_model(base_model, lora_cfg)
        ref_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
        # Freeze ref model
        ref_model.requires_grad_(False)

        self.trainer = PPOTrainer(
            config=ppo_cfg,
            model=policy_model,
            ref_model=ref_model,
            tokenizer=tokenizer,
        )
        self.tokenizer = tokenizer
        self.device = device
        self.policy_model: PeftModel = policy_model

    # ---------- utility helpers ----------
    async def save_lora_adapter(self, save_dir: str) -> str:
        """Save only LoRA weights for hot-reload."""
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        self.policy_model.save_pretrained(save_dir)
        logger.info("LoRA adapter saved", path=save_dir)
        return save_dir


class RLHFPPOTrainer:
    """Main RLHF PPO trainer orchestrating nightly runs."""

    def __init__(self, config):
        self.config = config
        self.ppo_cfg_raw = config.training.ppo
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            config.model.name, use_fast=True
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Build TRL PPOConfig
        self.trl_cfg = TRLPPOConfig(
            model_name=config.model.name,
            learning_rate=self.ppo_cfg_raw.learning_rate,
            batch_size=self.ppo_cfg_raw.batch_size,
            mini_batch_size=self.ppo_cfg_raw.mini_batch_size,
            gradient_accumulation_steps=self.ppo_cfg_raw.gradient_accumulation_steps,
            optimize_cuda_cache=True,
            kl_penalty="kl",
            target_kl=self.ppo_cfg_raw.target_kl,
            adap_kl_ctrl=True,
        )
        # Build LoRA config (fixed – could also pull from training.lora)
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.training.lora.rank,
            lora_alpha=self.config.training.lora.alpha,
            lora_dropout=self.config.training.lora.dropout,
            target_modules=self.config.training.lora.target_modules,
        )
        self.wrapper = PPOTrainerWrapper(
            base_model_name=config.model.name,
            tokenizer=self.tokenizer,
            ppo_cfg=self.trl_cfg,
            lora_cfg=lora_cfg,
        )
        self.running = False
        self.metrics: Dict[str, Any] = {
            "steps": 0,
            "examples_processed": 0,
            "last_checkpoint": None,
        }

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    async def initialize(self) -> bool:
        logger.info("Initializing RLHF PPO trainer")
        self.running = True
        # Potentially load previous checkpoint – omitted for brevity
        return True

    async def shutdown(self):
        logger.info("Shutting down RLHF PPO trainer")
        self.running = False

    async def train_epoch(self, feedback_dataset: List[Dict[str, str]]):
        """Run one PPO epoch over a feedback dataset.

        Each item in *feedback_dataset* must contain keys ::
            {"prompt": str, "response": str, "score": float}
        """
        if not feedback_dataset:
            logger.warning("Empty feedback dataset, skipping PPO epoch")
            return

        logger.info("Starting PPO epoch", examples=len(feedback_dataset))
        batch_prompts = [item["prompt"] for item in feedback_dataset]
        batch_responses = [item["response"] for item in feedback_dataset]

        # Tokenise queries
        query_tensors = [
            self.tokenizer(p, return_tensors="pt").input_ids.squeeze(0).to(self.device)
            for p in batch_prompts
        ]

        # Tokenise responses (convert to tensors)
        response_tensors = [
            self.tokenizer(r, return_tensors="pt").input_ids.squeeze(0).to(self.device)
            for r in batch_responses
        ]

        # Compute reward signal (simple mapping of feedback score [-1,1]→reward); can be replaced with
        # separate reward model later.
        rewards = torch.tensor(
            [item["score"] for item in feedback_dataset], dtype=torch.float32, device=self.device
        )

        # PPO step
        stats = self.wrapper.trainer.step(query_tensors, response_tensors, rewards)
        self.metrics["steps"] += 1
        self.metrics["examples_processed"] += len(feedback_dataset)
        logger.info("PPO step completed", stats=stats)

    # ------------------------------------------------------------------
    async def save_checkpoint(self) -> str:
        ts = int(time.time())
        out_dir = f"ppo_adapter_{ts}"
        await self.wrapper.save_lora_adapter(out_dir)
        self.metrics["last_checkpoint"] = out_dir
        return out_dir

    def get_status(self) -> Dict[str, Any]:
        return {
            "running": self.running,
            "metrics": self.metrics,
        }

async def run_ppo_training_cycle(config_path: Optional[str] = None):
    """Simulates a PPO training cycle for standalone execution."""
    from config import load_config # Local import to avoid circular deps at module level

    logger.info("Starting PPO training cycle (standalone execution)...")
    config = load_config(config_path)

    trainer = RLHFPPOTrainer(config)
    await trainer.initialize()

    # Simulate fetching feedback data (replace with actual data loading)
    sample_feedback_data = [
        {"prompt": "What is the capital of France?", "response": "Paris is the capital.", "score": 1.0},
        {"prompt": "Explain black holes.", "response": "They are dense.", "score": 0.5},
        # Add more diverse examples for a real run
    ]

    if trainer.running:
        try:
            await trainer.train_epoch(sample_feedback_data)
            checkpoint_dir = await trainer.save_checkpoint()
            logger.info("PPO training cycle finished.", checkpoint=checkpoint_dir, status=trainer.get_status())
        except Exception as e:
            logger.error("Error during PPO training cycle", error=str(e))
        finally:
            await trainer.shutdown()
    else:
        logger.warning("PPO trainer did not initialize correctly.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run PPO RLHF Training Cycle.")
    parser.add_argument(
        "--config", 
        type=str, 
        default=None,  # Let load_config handle default
        help="Path to the configuration YAML file."
    )
    args = parser.parse_args()
    
    asyncio.run(run_ppo_training_cycle(args.config)) 