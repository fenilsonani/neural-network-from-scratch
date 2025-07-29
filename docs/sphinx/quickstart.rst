Quick Start Guide
================

This guide will get you up and running with Neural Architecture in just a few minutes.

Installation
-----------

.. code-block:: bash

   pip install neural-arch

Or install from source:

.. code-block:: bash

   git clone https://github.com/your-repo/neural-arch.git
   cd neural-arch
   pip install -e .

Your First Neural Network
-------------------------

Here's a complete example of creating and training a simple neural network:

.. code-block:: python

   import neural_arch as na
   import numpy as np
   
   # Create training data (XOR problem)
   X = na.Tensor([[0, 0], [0, 1], [1, 0], [1, 1]], requires_grad=True)
   y = na.Tensor([[0], [1], [1], [0]], requires_grad=True)
   
   # Define a simple neural network
   class XORNet:
       def __init__(self):
           self.fc1 = na.Linear(2, 4)
           self.fc2 = na.Linear(4, 1)
       
       def forward(self, x):
           x = na.relu(self.fc1(x))
           x = na.sigmoid(self.fc2(x))
           return x
       
       def parameters(self):
           params = {}
           params.update(self.fc1.parameters())
           params.update(self.fc2.parameters())
           return params
   
   # Create model and optimizer
   model = XORNet()
   optimizer = na.Adam(model.parameters(), lr=0.01)
   
   # Training loop
   for epoch in range(1000):
       # Forward pass
       predictions = model.forward(X)
       loss = na.mse_loss(predictions, y)
       
       # Backward pass
       loss.backward()
       optimizer.step()
       optimizer.zero_grad()
       
       # Print progress
       if epoch % 100 == 0:
           print(f"Epoch {epoch}, Loss: {loss.data:.4f}")
   
   # Test the trained model
   test_predictions = model.forward(X)
   print("Final predictions:")
   for i, pred in enumerate(test_predictions.data):
       print(f"Input: {X.data[i]}, Target: {y.data[i][0]}, Predicted: {pred[0]:.4f}")

Key Concepts
-----------

Tensors
~~~~~~~

Tensors are the fundamental data structure in Neural Architecture:

.. code-block:: python

   # Create tensors
   x = na.Tensor([1, 2, 3], requires_grad=True)
   y = na.Tensor([[1, 2], [3, 4]], requires_grad=True)
   
   # Operations create computational graphs
   z = na.add(x[0:2], na.Tensor([1, 1]))
   
   # Compute gradients
   loss = na.mean_pool(z)
   loss.backward()
   print(f"Gradients: {x.grad}")

Neural Network Layers
~~~~~~~~~~~~~~~~~~~~

Build networks with modular components:

.. code-block:: python

   # Linear transformations
   linear = na.Linear(input_size=10, output_size=5)
   
   # Activation functions
   activation = na.ReLU()
   
   # Apply layers
   x = na.Tensor(np.random.randn(32, 10), requires_grad=True)
   output = activation(linear(x))
   print(f"Output shape: {output.shape}")

Optimizers
~~~~~~~~~

Update model parameters during training:

.. code-block:: python

   # Create optimizer
   optimizer = na.Adam(model.parameters(), lr=0.001)
   
   # Training step
   loss = compute_loss(model, data)
   loss.backward()
   optimizer.step()
   optimizer.zero_grad()

Common Patterns
--------------

Classification Model
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class Classifier:
       def __init__(self, input_size, hidden_size, num_classes):
           self.fc1 = na.Linear(input_size, hidden_size)
           self.fc2 = na.Linear(hidden_size, num_classes)
       
       def forward(self, x):
           x = na.relu(self.fc1(x))
           x = na.softmax(self.fc2(x))
           return x
       
       def parameters(self):
           params = {}
           params.update(self.fc1.parameters())
           params.update(self.fc2.parameters())
           return params
   
   # Usage
   model = Classifier(784, 128, 10)
   optimizer = na.Adam(model.parameters(), lr=0.001)

Text Processing with Embeddings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class TextClassifier:
       def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
           self.embedding = na.Embedding(vocab_size, embed_dim)
           self.fc1 = na.Linear(embed_dim, hidden_dim)
           self.fc2 = na.Linear(hidden_dim, num_classes)
       
       def forward(self, token_ids):
           # token_ids shape: (batch_size, sequence_length)
           embedded = self.embedding(token_ids)  # (batch_size, seq_len, embed_dim)
           
           # Simple pooling over sequence dimension
           pooled = na.mean_pool(embedded, axis=1)  # (batch_size, embed_dim)
           
           # Classification layers
           hidden = na.relu(self.fc1(pooled))
           output = na.softmax(self.fc2(hidden))
           return output
   
   # Usage
   vocab_size = 10000
   model = TextClassifier(vocab_size, 128, 64, 2)

Transformer Model
~~~~~~~~~~~~~~~~

.. code-block:: python

   class SimpleTransformer:
       def __init__(self, vocab_size, d_model, num_heads, num_layers):
           self.embedding = na.Embedding(vocab_size, d_model)
           self.layers = [na.TransformerBlock(d_model, num_heads, d_model * 4) 
                         for _ in range(num_layers)]
           self.output = na.Linear(d_model, vocab_size)
       
       def forward(self, token_ids):
           x = self.embedding(token_ids)
           
           for layer in self.layers:
               x = layer(x)
           
           return na.softmax(self.output(x))
       
       def parameters(self):
           params = {}
           params.update(self.embedding.parameters())
           
           for i, layer in enumerate(self.layers):
               layer_params = layer.parameters()
               for name, param in layer_params.items():
                   params[f'layer_{i}_{name}'] = param
           
           params.update(self.output.parameters())
           return params

Next Steps
---------

Now that you've seen the basics, explore these areas:

1. **Advanced Features**: Learn about configuration management, CLI tools, and performance optimization
2. **Examples**: Check out complete examples in the ``examples/`` directory
3. **API Reference**: Dive deep into the full API documentation
4. **Performance Guide**: Optimize your models for production use

Common Issues
------------

**Import Errors**
   Make sure Neural Architecture is properly installed: ``pip install neural-arch``

**Shape Mismatches**
   Check tensor shapes when debugging: ``print(tensor.shape)``

**Gradient Issues**
   Ensure ``requires_grad=True`` for tensors that need gradients

**Memory Issues**
   Use ``tensor.detach()`` to break gradient computation when not needed

**Performance**
   Use batch processing and avoid Python loops over tensor operations

Getting Help
-----------

- **Documentation**: Full API reference and guides
- **Examples**: Complete working examples
- **Issues**: Report bugs and request features on GitHub
- **Community**: Join discussions and get help from other users

Ready to build something amazing? Check out the tutorial for more detailed examples!