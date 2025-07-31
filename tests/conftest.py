"""
Pytest configuration file for Amharic H-Net v2 tests.
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, Any

import pytest
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer


@pytest.fixture(scope="session")
def test_config() -> Dict[str, Any]:
    """Provide test configuration."""
    return {
        "model": {
            "name": "amharic-hnet-test",
            "vocab_size": 1000,
            "hidden_size": 128,
            "num_layers": 2,
            "num_heads": 4,
            "max_seq_length": 256,
        },
        "training": {
            "batch_size": 4,
            "learning_rate": 1e-4,
            "num_epochs": 2,
            "warmup_steps": 10,
        },
        "data": {
            "max_length": 256,
            "morpheme_ratio": 0.15,
        }
    }


@pytest.fixture(scope="session")
def device():
    """Provide device for testing (prefer CPU for CI)."""
    if os.environ.get("CI") or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda:0")


@pytest.fixture
def temp_dir():
    """Provide temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_amharic_text():
    """Provide sample Amharic text for testing."""
    return [
        "እንደምን አደርክ? ጤና ይስጥልኝ።",
        "አዲስ አበባ የኢትዮጵያ ዋና ከተማ ናት።",
        "በዚህ ዓመት ብዙ ለውጦች ይኖራሉ።",
        "የአማርኛ ቋንቋ ውብ እና ሰፊ ነው።",
        "ተማሪዎች በጠንካራ እየተማሩ ነው።"
    ]


@pytest.fixture
def sample_morpheme_data():
    """Provide sample morpheme-annotated data."""
    return [
        {
            "text": "እንደምን አደርክ?",
            "morphemes": ["እንደ", "ምን", "አ", "ደር", "ክ", "?"],
            "tags": ["PREP", "PRON", "PREP", "VERB", "SUBJ", "PUNCT"]
        },
        {
            "text": "ጤና ይስጥልኝ።",
            "morphemes": ["ጤና", "ይ", "ስጥ", "ልኝ", "።"],
            "tags": ["NOUN", "SUBJ", "VERB", "OBJ", "PUNCT"]
        }
    ]


@pytest.fixture
def mock_tokenizer():
    """Provide a mock tokenizer for testing."""
    class MockTokenizer:
        def __init__(self):
            self.vocab_size = 1000
            self.pad_token_id = 0
            self.unk_token_id = 1
            self.cls_token_id = 2
            self.sep_token_id = 3
            self.mask_token_id = 4
            
        def encode(self, text, max_length=None, padding=False, truncation=False):
            # Simple mock encoding
            tokens = [self.cls_token_id] + [hash(char) % 1000 for char in text] + [self.sep_token_id]
            if max_length and truncation:
                tokens = tokens[:max_length]
            if max_length and padding:
                tokens.extend([self.pad_token_id] * (max_length - len(tokens)))
            return tokens
            
        def decode(self, tokens):
            return "mock_decoded_text"
            
        def __call__(self, text, **kwargs):
            return {"input_ids": self.encode(text, **kwargs)}
    
    return MockTokenizer()


@pytest.fixture
def sample_dataset(sample_amharic_text, temp_dir):
    """Create a sample dataset file for testing."""
    dataset_file = temp_dir / "sample_dataset.txt"
    with open(dataset_file, "w", encoding="utf-8") as f:
        for text in sample_amharic_text:
            f.write(text + "\n")
    return dataset_file


@pytest.fixture
def model_checkpoint_path(temp_dir):
    """Provide path for model checkpoints during testing."""
    checkpoint_dir = temp_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    return checkpoint_dir


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set random seeds for reproducible tests."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def training_config(temp_dir):
    """Provide training configuration for tests."""
    config = {
        "model": {
            "type": "hnet_amharic",
            "hidden_size": 128,
            "num_layers": 2,
            "num_heads": 4,
            "vocab_size": 1000,
        },
        "training": {
            "batch_size": 2,
            "learning_rate": 5e-4,
            "num_epochs": 1,
            "save_steps": 50,
            "eval_steps": 50,
            "logging_steps": 10,
        },
        "data": {
            "max_length": 128,
            "morpheme_masking_prob": 0.15,
        },
        "output": {
            "output_dir": str(temp_dir / "outputs"),
            "logging_dir": str(temp_dir / "logs"),
        }
    }
    return config


# Skip markers for different test types
def requires_gpu():
    """Skip test if GPU is not available."""
    return pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="GPU not available"
    )


def requires_large_memory():
    """Skip test if insufficient memory."""
    return pytest.mark.skipif(
        torch.cuda.get_device_properties(0).total_memory < 8e9 if torch.cuda.is_available() else True,
        reason="Insufficient GPU memory"
    )


def slow_test():
    """Mark test as slow."""
    return pytest.mark.slow


def integration_test():
    """Mark test as integration test."""
    return pytest.mark.integration


# Custom pytest hooks
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "gpu: marks tests requiring GPU")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Add unit marker to tests not marked as integration
        if not any(marker.name == "integration" for marker in item.iter_markers()):
            item.add_marker(pytest.mark.unit)
        
        # Add slow marker to tests with "slow" in name
        if "slow" in item.name.lower():
            item.add_marker(pytest.mark.slow)