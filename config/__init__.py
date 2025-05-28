"""Configuration management for neuroplastic Qwen system."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class ModelConfig:
    name: str
    device: str
    dtype: str
    max_model_len: int
    trust_remote_code: bool


@dataclass
class ServingConfig:
    engine: str
    host: str
    port: int
    workers: int
    gpu_memory_utilization: float
    max_num_batched_tokens: int
    max_num_seqs: int
    enable_lora: bool
    max_lora_rank: int
    max_loras: int


@dataclass
class APIConfig:
    host: str
    port: int
    log_level: str
    enable_tracing: bool
    max_request_size: int


@dataclass
class StorageConfig:
    type: str  # 's3', 'minio', 'gcs', 'azure', etc.
    bucket: str
    region: Optional[str] = None  # Required for S3, GCS, Azure
    prefix: str = ""
    endpoint_url: Optional[str] = None  # For S3-compatible like MinIO
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    local_cache_dir: str = "./cache"
    cache_max_size_gb: Optional[int] = 50
    cache_cleanup_interval_seconds: Optional[int] = 3600  # 1 hour
    cache_max_age_hours: Optional[int] = 24


@dataclass
class KafkaTopics:
    traces: str
    feedback: str
    ewc_samples: str
    tot_traces: str
    metrics: str # Added for training metrics


@dataclass
class KafkaConfig:
    bootstrap_servers: list
    topics: KafkaTopics
    consumer_group: str
    auto_offset_reset: str


@dataclass
class RedisConfig:
    host: str
    port: int
    db: int
    password: Optional[str] = None


@dataclass
class EWCConfig:
    # Parameters from sample_config.yaml, expected by EWCTrainer
    lambda_init: float
    lambda_max: float
    lambda_min: float
    lambda_decay: float
    lambda_adjustment_interval: float # Renamed from sample_config for consistency
    fisher_samples: int
    fisher_batch_size: int
    consolidation_threshold: float
    importance_threshold: float
    max_stored_tasks: int
    # Parameters from original base.yaml EWCConfig, potentially for future use or other trainers
    fisher_update_interval: Optional[int] = None # Kept as optional
    target_modules: Optional[list] = None # Kept as optional, EWCTrainer doesn't use it now


@dataclass
class Rank1Config:
    max_slices_per_layer: int
    update_threshold: int
    importance_decay: float


@dataclass
class PPOConfig:
    schedule: str
    batch_size: int
    learning_rate: float # Note: also in GeneralTrainingConfig, consider which one to use
    mini_batch_size: int
    gradient_accumulation_steps: int
    target_kl: float


@dataclass
class GeneralTrainingConfig:
    enabled: bool
    learning_rate: float
    max_queue_size: int
    queue_timeout_seconds: float
    max_workers: int
    positive_feedback_threshold: float
    negative_feedback_threshold: float
    lora_adaptation_threshold: float
    replay_optimization_interval_seconds: int
    ewc_adjustment_interval_seconds: int
    lora_cleanup_interval_seconds: int
    metrics_log_interval_seconds: int


@dataclass
class LoRAConfig:
    rank: int
    alpha: int
    dropout: float
    target_modules: list[str]
    max_adapters: int
    min_samples_for_adaptation: int
    adapter_merge_threshold: float
    dynamic_rank_enabled: bool
    rank_min: int
    rank_max: int


@dataclass
class ToTConfig:
    enabled: bool
    max_depth: int
    branching_factor: int
    evaluation_samples: int
    improvement_threshold: float
    max_optimization_time_seconds: int
    beam_width: int
    search_strategy: str
    pruning_enabled: bool
    pruning_threshold: float
    evaluation_metrics: list[str]
    metric_weights: list[float]


@dataclass
class TrainingConfig:
    general: GeneralTrainingConfig
    ewc: EWCConfig
    lora: LoRAConfig
    tot: ToTConfig
    rank1: Rank1Config
    ppo: PPOConfig


@dataclass
class HotReloadConfig:
    check_interval: int
    download_timeout: int
    validation_enabled: bool
    rollback_on_failure: bool


@dataclass
class MonitoringConfig:
    prometheus_port: int
    health_check_interval: int
    performance_threshold: float


@dataclass
class LoggingConfig:
    level: str
    format: str
    file: str
    max_size: str
    backup_count: int


@dataclass
class Config:
    model: ModelConfig
    serving: ServingConfig
    api: APIConfig
    storage: StorageConfig
    kafka: KafkaConfig
    redis: RedisConfig
    training: TrainingConfig
    hot_reload: HotReloadConfig
    monitoring: MonitoringConfig
    logging: LoggingConfig


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = os.getenv("CONFIG_PATH", "config/base.yaml")
    
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Apply environment variable overrides
    config_dict = _apply_env_overrides(config_dict)
    
    return _dict_to_config(config_dict)


def _apply_env_overrides(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Apply environment variable overrides to configuration."""
    # Model overrides
    if "MODEL_NAME" in os.environ:
        config_dict["model"]["name"] = os.environ["MODEL_NAME"]
    
    # Storage overrides
    if "S3_BUCKET" in os.environ:
        config_dict["storage"]["bucket"] = os.environ["S3_BUCKET"]
    if "S3_REGION" in os.environ:
        config_dict["storage"]["region"] = os.environ["S3_REGION"]
    if "S3_ENDPOINT_URL" in os.environ:
        config_dict["storage"]["endpoint_url"] = os.environ["S3_ENDPOINT_URL"]
    if "AWS_ACCESS_KEY_ID" in os.environ:
        config_dict["storage"]["aws_access_key_id"] = os.environ["AWS_ACCESS_KEY_ID"]
    if "AWS_SECRET_ACCESS_KEY" in os.environ:
        config_dict["storage"]["aws_secret_access_key"] = os.environ["AWS_SECRET_ACCESS_KEY"]
    
    # Kafka overrides
    if "KAFKA_BOOTSTRAP_SERVERS" in os.environ:
        servers = os.environ["KAFKA_BOOTSTRAP_SERVERS"].split(",")
        config_dict["kafka"]["bootstrap_servers"] = servers
    
    # Redis overrides
    if "REDIS_HOST" in os.environ:
        config_dict["redis"]["host"] = os.environ["REDIS_HOST"]
    if "REDIS_PASSWORD" in os.environ:
        config_dict["redis"]["password"] = os.environ["REDIS_PASSWORD"]
    
    return config_dict


def _dict_to_config(config_dict: Dict[str, Any]) -> Config:
    """Convert dictionary to Config dataclass."""
    return Config(
        model=ModelConfig(**config_dict["model"]),
        serving=ServingConfig(**config_dict["serving"]),
        api=APIConfig(**config_dict["api"]),
        storage=StorageConfig(**config_dict["storage"]),
        kafka=KafkaConfig(
            bootstrap_servers=config_dict["kafka"]["bootstrap_servers"],
            topics=KafkaTopics(**config_dict["kafka"]["topics"]),
            consumer_group=config_dict["kafka"]["consumer_group"],
            auto_offset_reset=config_dict["kafka"]["auto_offset_reset"]
        ),
        redis=RedisConfig(**config_dict["redis"]),
        training=TrainingConfig(
            general=GeneralTrainingConfig(**config_dict["training"]["general"]),
            ewc=EWCConfig(**config_dict["training"]["ewc"]),
            lora=LoRAConfig(**config_dict["training"]["lora"]),
            tot=ToTConfig(**config_dict["training"]["tot"]),
            rank1=Rank1Config(**config_dict["training"]["rank1"]),
            ppo=PPOConfig(**config_dict["training"]["ppo"])
        ),
        hot_reload=HotReloadConfig(**config_dict["hot_reload"]),
        monitoring=MonitoringConfig(**config_dict["monitoring"]),
        logging=LoggingConfig(**config_dict["logging"])
    ) 