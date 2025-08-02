"""
Pytest configuration for neural architecture tests.
"""

import pytest
import numpy as np


@pytest.fixture
def simple_tensor():
    """Create a simple tensor for testing."""
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
    from neural_arch import Tensor
    return Tensor([1, 2, 3], requires_grad=True)


@pytest.fixture
def simple_matrix():
    """Create a simple matrix tensor for testing."""
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
    from neural_arch import Tensor
    return Tensor([[1, 2], [3, 4]], requires_grad=True)


@pytest.fixture
def simple_linear_layer():
    """Create a simple linear layer for testing."""
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
    from neural_arch import Linear
    return Linear(2, 1)


@pytest.fixture
def simple_embedding():
    """Create a simple embedding layer for testing."""
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
    from neural_arch import Embedding
    return Embedding(vocab_size=5, embed_dim=3)


@pytest.fixture
def adam_optimizer():
    """Create Adam optimizer for testing."""
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
    from neural_arch import Tensor, Adam
    
    params = {
        'w': Tensor([[1.0, 2.0]], requires_grad=True),
        'b': Tensor([0.5], requires_grad=True)
    }
    return Adam(params, lr=0.01)


@pytest.fixture
def sample_text_data():
    """Create sample text data for training tests."""
    return "hello world test neural networks"


# Set random seed for reproducible tests
np.random.seed(42)