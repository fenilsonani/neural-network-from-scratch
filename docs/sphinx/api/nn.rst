Neural Network Module
====================

The nn module provides high-level neural network layers and components for building deep learning models.

.. currentmodule:: neural_arch.nn

Linear Layers
------------

.. autoclass:: Linear
   :members:
   :special-members: __init__, __call__
   :show-inheritance:

Embedding Layers
---------------

.. autoclass:: Embedding
   :members:
   :special-members: __init__, __call__
   :show-inheritance:

Activation Layers
----------------

.. autoclass:: ReLU
   :members:
   :special-members: __init__, __call__
   :show-inheritance:

.. autoclass:: Softmax
   :members:
   :special-members: __init__, __call__
   :show-inheritance:

.. autoclass:: Sigmoid
   :members:
   :special-members: __init__, __call__
   :show-inheritance:

.. autoclass:: Tanh
   :members:
   :special-members: __init__, __call__
   :show-inheritance:

.. autoclass:: GELU
   :members:
   :special-members: __init__, __call__
   :show-inheritance:

Normalization Layers
-------------------

.. autoclass:: LayerNorm
   :members:
   :special-members: __init__, __call__
   :show-inheritance:

Attention Mechanisms
-------------------

.. autoclass:: MultiHeadAttention
   :members:
   :special-members: __init__, __call__
   :show-inheritance:

Transformer Components
---------------------

.. autoclass:: TransformerBlock
   :members:
   :special-members: __init__, __call__
   :show-inheritance:

.. autoclass:: TransformerDecoderBlock
   :members:
   :special-members: __init__, __call__
   :show-inheritance:

Examples
--------

Building a Simple Neural Network
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import neural_arch as na
   
   class SimpleClassifier:
       def __init__(self, input_size, hidden_size, num_classes):
           self.fc1 = na.Linear(input_size, hidden_size)
           self.relu = na.ReLU()
           self.fc2 = na.Linear(hidden_size, num_classes)
           self.softmax = na.Softmax()
       
       def forward(self, x):
           x = self.fc1(x)
           x = self.relu(x)
           x = self.fc2(x)
           return self.softmax(x)
       
       def parameters(self):
           params = {}
           params.update(self.fc1.parameters())
           params.update(self.fc2.parameters())
           return params
   
   # Create and use the model
   model = SimpleClassifier(784, 128, 10)
   inputs = na.Tensor(np.random.randn(32, 784), requires_grad=True)
   outputs = model.forward(inputs)

Using Embedding Layers
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # For text processing
   vocab_size = 10000
   embed_dim = 300
   
   embedding = na.Embedding(vocab_size, embed_dim)
   
   # Input: batch of token sequences
   token_ids = np.array([[1, 5, 3, 7], [2, 4, 8, 9]])  # Shape: (2, 4)
   embedded = embedding(token_ids)  # Shape: (2, 4, 300)

Multi-Head Attention Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Transformer-style attention
   d_model = 512
   num_heads = 8
   
   attention = na.MultiHeadAttention(d_model, num_heads)
   
   # Input: (batch_size, sequence_length, d_model)
   x = na.Tensor(np.random.randn(32, 100, d_model), requires_grad=True)
   attended = attention(x)  # Same shape as input

Building a Transformer Block
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Complete transformer block
   d_model = 512
   num_heads = 8
   d_ff = 2048
   
   transformer_block = na.TransformerBlock(d_model, num_heads, d_ff)
   
   # Process sequence
   sequence = na.Tensor(np.random.randn(16, 50, d_model), requires_grad=True)
   output = transformer_block(sequence)

Layer Normalization
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Normalize features
   batch_size, seq_len, d_model = 32, 100, 512
   
   layer_norm = na.LayerNorm(d_model)
   x = na.Tensor(np.random.randn(batch_size, seq_len, d_model), requires_grad=True)
   
   # Normalize along the last dimension
   normalized = layer_norm(x)
   
   # Check normalization properties
   mean_vals = np.mean(normalized.data, axis=-1)  # Should be ~0
   std_vals = np.std(normalized.data, axis=-1)    # Should be ~1

Training a Neural Network
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Complete training example
   import neural_arch as na
   
   # Define model
   class MLP:
       def __init__(self):
           self.layers = [
               na.Linear(784, 256),
               na.ReLU(),
               na.Linear(256, 128), 
               na.ReLU(),
               na.Linear(128, 10),
               na.Softmax()
           ]
       
       def forward(self, x):
           for layer in self.layers:
               x = layer(x)
           return x
       
       def parameters(self):
           params = {}
           for i, layer in enumerate(self.layers):
               if hasattr(layer, 'parameters'):
                   layer_params = layer.parameters()
                   for name, param in layer_params.items():
                       params[f'layer_{i}_{name}'] = param
           return params
   
   # Training setup
   model = MLP()
   optimizer = na.Adam(model.parameters(), lr=0.001)
   
   # Training loop
   for epoch in range(100):
       # Forward pass
       predictions = model.forward(inputs)
       loss = na.cross_entropy_loss(predictions, targets)
       
       # Backward pass
       loss.backward()
       optimizer.step()
       optimizer.zero_grad()
       
       if epoch % 10 == 0:
           print(f"Epoch {epoch}, Loss: {loss.data}")

Design Patterns
--------------

**Module Pattern**: All neural network components inherit from the base Module class, providing consistent interfaces for parameters and computation.

**Functional Core**: Layers are thin wrappers around functional operations, maintaining separation between computation and state.

**Gradient Flow**: All layers are designed to properly propagate gradients through the computational graph.

**Parameter Management**: Consistent parameter naming and access patterns across all layers.

Performance Considerations
-------------------------

- **Memory Efficiency**: Layers minimize memory allocations during forward and backward passes
- **Gradient Computation**: Efficient backpropagation through all layer types
- **Numerical Stability**: Careful implementation of normalization and activation functions
- **Batch Processing**: All layers support batch processing for efficient training