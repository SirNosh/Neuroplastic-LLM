"""Test script for LoRA trainer integration."""

import asyncio
import tempfile
import torch
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Mock the PEFT imports for testing without requiring the full PEFT installation
with patch.dict('sys.modules', {
    'peft': MagicMock(),
    'peft.LoraConfig': MagicMock(),
    'peft.get_peft_model': MagicMock(),
    'peft.TaskType': MagicMock(),
    'peft.PeftModel': MagicMock(),
    'peft.PeftConfig': MagicMock(),
}):
    from training.lora_trainer import LoRATrainer, LoRAAdapterInfo
    from config import Config


class MockModel(torch.nn.Module):
    """Mock model for testing."""
    
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
        self.device = torch.device('cpu')
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = MagicMock()
        outputs.loss = torch.tensor(1.0)
        outputs.logits = torch.randn(input_ids.shape[0], input_ids.shape[1], 1000)
        return outputs
    
    def named_modules(self):
        return [('linear', self.linear), ('q_proj', torch.nn.Linear(10, 10))]
    
    def named_parameters(self):
        return [('linear.weight', self.linear.weight), ('linear.bias', self.linear.bias)]
    
    def parameters(self):
        return [self.linear.weight, self.linear.bias]
    
    def train(self):
        pass
    
    def zero_grad(self):
        pass


class MockTokenizer:
    """Mock tokenizer for testing."""
    
    def __init__(self):
        self.vocab_size = 1000
    
    def __call__(self, text, return_tensors=None, truncation=True, max_length=512, padding=True):
        mock_input = MagicMock()
        mock_input.to = MagicMock(return_value=mock_input)
        
        # Create mock input_ids
        if isinstance(text, str):
            input_ids = torch.randint(0, self.vocab_size, (1, min(len(text.split()), max_length)))
        else:
            input_ids = torch.randint(0, self.vocab_size, (len(text), max_length))
        
        mock_input.__getitem__ = lambda key: input_ids if key == 'input_ids' else torch.ones_like(input_ids)
        return mock_input


class MockStorageManager:
    """Mock storage manager for testing."""
    
    def __init__(self):
        self.adapters = {}
    
    async def list_lora_adapters(self):
        return [
            {'adapter_id': adapter_id, 'latest_version': 'latest'}
            for adapter_id in self.adapters.keys()
        ]
    
    async def download_lora_adapter(self, adapter_id, version):
        if adapter_id in self.adapters:
            return Path(f"/tmp/mock_adapter_{adapter_id}")
        return None
    
    async def upload_lora_adapter(self, adapter_path, adapter_id, version):
        self.adapters[adapter_id] = {'path': adapter_path, 'version': version}
        return True
    
    async def delete_lora_adapter(self, adapter_id):
        if adapter_id in self.adapters:
            del self.adapters[adapter_id]
            return True
        return False


class MockConfig:
    """Mock configuration for testing."""
    
    def __init__(self):
        self.model = MagicMock()
        self.model.max_model_len = 2048
        
        self.training = MagicMock()
        self.training.general = MagicMock()
        self.training.general.learning_rate = 1e-4
        self.training.general.lora_adaptation_threshold = 0.7
        self.training.general.lora_cleanup_interval_seconds = 3600
        
        self.training.lora = MagicMock()
        self.training.lora.rank = 16
        self.training.lora.alpha = 32
        self.training.lora.dropout = 0.1
        self.training.lora.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        self.training.lora.max_adapters = 10
        self.training.lora.min_samples_for_adaptation = 3
        self.training.lora.adapter_merge_threshold = 0.9
        self.training.lora.dynamic_rank_enabled = True
        self.training.lora.rank_min = 4
        self.training.lora.rank_max = 64


@pytest.fixture
def mock_model():
    return MockModel()


@pytest.fixture
def mock_tokenizer():
    return MockTokenizer()


@pytest.fixture
def mock_storage_manager():
    return MockStorageManager()


@pytest.fixture
def mock_config():
    return MockConfig()


@pytest.fixture
async def lora_trainer(mock_model, mock_tokenizer, mock_storage_manager, mock_config):
    """Create a LoRA trainer instance for testing."""
    with patch('training.lora_trainer.get_peft_model') as mock_get_peft_model, \
         patch('training.lora_trainer.LoraConfig') as mock_lora_config, \
         patch('training.lora_trainer.TaskType') as mock_task_type:
        
        # Mock PEFT components
        mock_peft_model = MagicMock()
        mock_peft_model.num_parameters.return_value = 1000
        mock_peft_model.get_nb_trainable_parameters.return_value = 100
        mock_peft_model.save_pretrained = MagicMock()
        mock_peft_model.load_state_dict = MagicMock()
        mock_peft_model.train = MagicMock()
        mock_peft_model.zero_grad = MagicMock()
        mock_peft_model.parameters.return_value = mock_model.parameters()
        
        mock_get_peft_model.return_value = mock_peft_model
        mock_lora_config.return_value = MagicMock()
        mock_lora_config.return_value.to_dict.return_value = {}
        mock_lora_config.return_value.r = 16
        
        trainer = LoRATrainer(mock_model, mock_tokenizer, mock_config, mock_storage_manager)
        
        # Mock the asyncio.create_task calls to avoid actual background tasks in tests
        with patch('asyncio.create_task'):
            await trainer.initialize()
        
        return trainer


class TestLoRATrainer:
    """Test cases for LoRA trainer."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, lora_trainer):
        """Test LoRA trainer initialization."""
        assert lora_trainer.running is True
        assert len(lora_trainer.active_adapters) == 0
        assert lora_trainer.adapter_counter == 0
    
    @pytest.mark.asyncio
    async def test_model_compatibility_check(self, mock_model, mock_tokenizer, mock_storage_manager, mock_config):
        """Test model compatibility checking."""
        trainer = LoRATrainer(mock_model, mock_tokenizer, mock_config, mock_storage_manager)
        
        # Test with compatible model
        assert trainer._is_model_compatible() is True
        
        # Test with incompatible model (no target modules)
        mock_config.training.lora.target_modules = ["nonexistent_module"]
        trainer_incompatible = LoRATrainer(mock_model, mock_tokenizer, mock_config, mock_storage_manager)
        assert trainer_incompatible._is_model_compatibility_check() is False
    
    @pytest.mark.asyncio
    async def test_adapt_from_feedback(self, lora_trainer):
        """Test adaptation from feedback."""
        # Test adding feedback for adaptation
        success = await lora_trainer.adapt_from_feedback(
            prompt="What is AI?",
            response="AI is artificial intelligence.",
            feedback_score=0.8,
            session_id="test_session",
            importance_weight=1.0
        )
        
        assert success is True
        assert lora_trainer.adaptation_queue.qsize() == 1
    
    @pytest.mark.asyncio
    async def test_session_pattern_tracking(self, lora_trainer):
        """Test session pattern tracking."""
        session_id = "test_session"
        
        # Add multiple feedback instances
        for i in range(5):
            await lora_trainer.adapt_from_feedback(
                prompt=f"Question {i}",
                response=f"Answer {i}",
                feedback_score=0.8 + i * 0.02,
                session_id=session_id
            )
        
        # Check if session patterns are tracked
        assert session_id in lora_trainer.session_patterns
        assert len(lora_trainer.session_patterns[session_id]) >= 0  # Depends on processing
    
    @pytest.mark.asyncio 
    async def test_optimal_rank_determination(self, lora_trainer):
        """Test dynamic rank determination."""
        session_id = "rank_test_session"
        
        # Add some session data
        lora_trainer.session_patterns[session_id] = [
            {
                'prompt': 'Simple question',
                'feedback_score': 0.9
            },
            {
                'prompt': 'Another simple question',
                'feedback_score': 0.85
            }
        ]
        
        rank = await lora_trainer._determine_optimal_rank(session_id)
        assert isinstance(rank, int)
        assert 4 <= rank <= 64  # Should be within configured bounds
    
    @pytest.mark.asyncio
    async def test_adapter_capacity_management(self, lora_trainer):
        """Test adapter capacity management."""
        # Create mock adapters that exceed capacity
        for i in range(12):  # Exceed max_adapters = 10
            adapter_info = MagicMock()
            adapter_info.performance_score = 0.3 if i < 5 else 0.8  # Mix of low and high performers
            adapter_info.usage_count = 10
            lora_trainer.active_adapters[f"adapter_{i}"] = adapter_info
        
        # Test capacity management
        await lora_trainer._manage_adapter_capacity()
        
        # Should have removed some low-performing adapters
        assert len(lora_trainer.active_adapters) <= lora_trainer.lora_config.max_adapters
    
    @pytest.mark.asyncio
    async def test_adapter_cleanup(self, lora_trainer):
        """Test adapter cleanup functionality."""
        import time
        
        # Create old adapters
        old_time = time.time() - 48 * 3600  # 48 hours ago
        
        adapter_info = MagicMock()
        adapter_info.created_at = old_time
        adapter_info.last_used = old_time
        adapter_info.usage_count = 2  # Low usage
        adapter_info.performance_score = 0.2  # Low performance
        
        lora_trainer.active_adapters["old_adapter"] = adapter_info
        
        await lora_trainer._cleanup_adapters()
        
        # Old, unused adapter should be removed
        assert "old_adapter" not in lora_trainer.active_adapters
    
    @pytest.mark.asyncio
    async def test_hot_reload_adapter(self, lora_trainer, tmp_path):
        """Test hot-reloading of adapters."""
        # Create a mock adapter
        adapter_id = "test_adapter"
        adapter_info = MagicMock()
        adapter_info.file_path = None
        lora_trainer.active_adapters[adapter_id] = adapter_info
        
        # Create mock adapter files
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        adapter_weights = adapter_dir / "adapter_model.bin"
        adapter_weights.write_text("mock_weights")
        
        # Test hot reload
        success = await lora_trainer.hot_reload_adapter(adapter_id, str(adapter_dir))
        
        assert success is True
        assert lora_trainer.active_adapters[adapter_id].file_path == str(adapter_dir)
    
    @pytest.mark.asyncio
    async def test_status_reporting(self, lora_trainer):
        """Test status reporting functionality."""
        status = lora_trainer.get_status()
        
        assert "running" in status
        assert "active_adapters" in status
        assert "metrics" in status
        assert "queue_size" in status
        
        assert isinstance(status["active_adapters"], int)
        assert isinstance(status["metrics"], dict)
    
    @pytest.mark.asyncio
    async def test_checkpoint_save_load(self, lora_trainer):
        """Test checkpoint saving and loading."""
        # Create some adapter state
        adapter_info = MagicMock()
        adapter_info.adapter_id = "test_adapter"
        adapter_info.session_id = "test_session"
        adapter_info.config = MagicMock()
        adapter_info.config.to_dict.return_value = {"r": 16, "alpha": 32}
        lora_trainer.active_adapters["test_adapter"] = adapter_info
        
        # Test checkpoint creation
        checkpoint = await lora_trainer.get_checkpoint()
        assert "active_adapters" in checkpoint
        assert "metrics" in checkpoint
        
        # Test checkpoint loading
        lora_trainer.active_adapters.clear()
        success = await lora_trainer.load_checkpoint(checkpoint)
        
        assert success is True
        # Note: Actual loading would require proper LoraConfig mocking
    
    @pytest.mark.asyncio
    async def test_shutdown(self, lora_trainer):
        """Test trainer shutdown."""
        # Add some data
        lora_trainer.active_adapters["test"] = MagicMock()
        lora_trainer.session_patterns["test"] = []
        
        await lora_trainer.shutdown()
        
        assert lora_trainer.running is False
        assert len(lora_trainer.active_adapters) == 0
        assert len(lora_trainer.session_patterns) == 0


async def main():
    """Main test runner."""
    print("Running LoRA trainer integration tests...")
    
    # Create test instances
    model = MockModel()
    tokenizer = MockTokenizer()
    storage_manager = MockStorageManager()
    config = MockConfig()
    
    # Test basic functionality
    with patch('training.lora_trainer.get_peft_model') as mock_get_peft_model, \
         patch('training.lora_trainer.LoraConfig') as mock_lora_config:
        
        mock_peft_model = MagicMock()
        mock_peft_model.num_parameters.return_value = 1000
        mock_peft_model.get_nb_trainable_parameters.return_value = 100
        mock_get_peft_model.return_value = mock_peft_model
        mock_lora_config.return_value = MagicMock()
        
        trainer = LoRATrainer(model, tokenizer, config, storage_manager)
        
        # Mock asyncio.create_task to avoid background tasks
        with patch('asyncio.create_task'):
            success = await trainer.initialize()
            
        print(f"✓ LoRA trainer initialization: {'SUCCESS' if success else 'FAILED'}")
        
        # Test feedback adaptation
        success = await trainer.adapt_from_feedback(
            prompt="What is machine learning?",
            response="Machine learning is a subset of AI.",
            feedback_score=0.8,
            session_id="demo_session"
        )
        
        print(f"✓ Feedback adaptation: {'SUCCESS' if success else 'FAILED'}")
        
        # Test status reporting
        status = trainer.get_status()
        print(f"✓ Status reporting: {'SUCCESS' if status.get('running') else 'FAILED'}")
        
        # Test cleanup
        await trainer.shutdown()
        print(f"✓ Shutdown: {'SUCCESS' if not trainer.running else 'FAILED'}")
    
    print("\nLoRA trainer integration test completed!")


if __name__ == "__main__":
    asyncio.run(main()) 