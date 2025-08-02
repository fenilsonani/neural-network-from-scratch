Tutorial
========

This comprehensive tutorial will guide you through building progressively more complex neural networks with Neural Architecture.

.. contents:: Table of Contents
   :local:
   :depth: 2

Part 1: Understanding Tensors
-----------------------------

Tensors are the foundation of Neural Architecture. Let's start with the basics:

Creating Tensors
~~~~~~~~~~~~~~~

.. code-block:: python

   import neural_arch as na
   import numpy as np
   
   # Create tensors from Python lists
   x = na.Tensor([1, 2, 3, 4])
   print(f"1D tensor: {x.data}")
   
   # Create 2D tensors (matrices)
   matrix = na.Tensor([[1, 2], [3, 4]])
   print(f"2D tensor shape: {matrix.shape}")
   
   # Create tensors from NumPy arrays
   numpy_data = np.random.randn(3, 4)
   tensor_from_numpy = na.Tensor(numpy_data)
   
   # Enable gradient computation
   x_grad = na.Tensor([1.0, 2.0, 3.0], requires_grad=True)
   print(f"Requires grad: {x_grad.requires_grad}")

Tensor Operations
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Basic arithmetic
   a = na.Tensor([[1, 2], [3, 4]], requires_grad=True)
   b = na.Tensor([[2, 1], [1, 2]], requires_grad=True)
   
   # Element-wise operations
   c = na.add(a, b)      # Addition
   d = na.mul(a, b)      # Element-wise multiplication
   e = na.matmul(a, b)   # Matrix multiplication
   
   print(f"Addition result: {c.data}")
   print(f"Matrix multiplication: {e.data}")

Understanding Gradients
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Simple gradient example
   x = na.Tensor([2.0], requires_grad=True)
   
   # Compute y = x^2
   y = na.mul(x, x)
   print(f"y = x^2 = {y.data}")
   
   # Compute gradients (dy/dx = 2x)
   y.backward()
   print(f"Gradient dy/dx = {x.grad}")  # Should be 4.0

Part 2: Building Your First Neural Network
------------------------------------------

Let's build a simple neural network to classify the XOR function:

Network Architecture
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class XORNetwork:
       def __init__(self):
           # Hidden layer: 2 inputs -> 4 hidden units
           self.fc1 = na.Linear(2, 4)
           
           # Output layer: 4 hidden -> 1 output
           self.fc2 = na.Linear(4, 1)
       
       def forward(self, x):
           # Apply first layer + ReLU activation
           hidden = na.relu(self.fc1(x))
           
           # Apply output layer + sigmoid activation
           output = na.sigmoid(self.fc2(hidden))
           
           return output
       
       def parameters(self):
           """Get all trainable parameters."""
           params = {}
           params.update(self.fc1.parameters())
           params.update(self.fc2.parameters())
           return params

Training Data
~~~~~~~~~~~~

.. code-block:: python

   # XOR truth table
   inputs = na.Tensor([
       [0, 0],  # XOR(0,0) = 0
       [0, 1],  # XOR(0,1) = 1
       [1, 0],  # XOR(1,0) = 1
       [1, 1]   # XOR(1,1) = 0
   ], requires_grad=True)
   
   targets = na.Tensor([
       [0],
       [1], 
       [1],
       [0]
   ], requires_grad=True)
   
   print("Training data prepared")
   print(f"Input shape: {inputs.shape}")
   print(f"Target shape: {targets.shape}")

Training Loop
~~~~~~~~~~~~

.. code-block:: python

   # Create model and optimizer
   model = XORNetwork()
   optimizer = na.Adam(model.parameters(), lr=0.01)
   
   # Training loop
   losses = []
   for epoch in range(2000):
       # Forward pass
       predictions = model.forward(inputs)
       
       # Compute loss (Mean Squared Error)
       loss = na.mse_loss(predictions, targets)
       
       # Backward pass
       loss.backward()
       
       # Update parameters
       optimizer.step()
       optimizer.zero_grad()
       
       # Record loss
       losses.append(loss.data)
       
       # Print progress
       if epoch % 200 == 0:
           print(f"Epoch {epoch:4d}, Loss: {loss.data:.6f}")
   
   print("Training completed!")

Testing the Model
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Test the trained model
   print("\nTesting XOR network:")
   print("Input -> Predicted | Target")
   print("-" * 30)
   
   with na.no_grad():  # Disable gradients for inference
       predictions = model.forward(inputs)
       
       for i in range(len(inputs.data)):
           input_vals = inputs.data[i]
           predicted = predictions.data[i][0]
           target = targets.data[i][0]
           
           print(f"{input_vals} -> {predicted:.4f} | {target}")

Part 3: Image Classification with MNIST
---------------------------------------

Now let's build a more complex model for handwritten digit recognition:

Data Preparation
~~~~~~~~~~~~~~~

.. code-block:: python

   # Simulated MNIST-like data preparation
   def prepare_mnist_data():
       # In practice, you'd load real MNIST data
       # Here we simulate with random data for demonstration
       
       batch_size = 128
       image_size = 784  # 28x28 flattened
       num_classes = 10
       
       # Simulate training data
       train_images = na.Tensor(
           np.random.randn(batch_size, image_size), 
           requires_grad=True
       )
       
       # Random class labels
       train_labels = na.Tensor(
           np.random.randint(0, num_classes, (batch_size,))
       )
       
       return train_images, train_labels
   
   # Load data
   X_train, y_train = prepare_mnist_data()
   print(f"Training data: {X_train.shape}")
   print(f"Training labels: {y_train.shape}")

CNN-style Architecture
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class MNISTClassifier:
       def __init__(self):
           # Multi-layer perceptron architecture
           self.fc1 = na.Linear(784, 512)
           self.fc2 = na.Linear(512, 256)
           self.fc3 = na.Linear(256, 128)
           self.output = na.Linear(128, 10)
       
       def forward(self, x):
           # Layer 1
           x = na.relu(self.fc1(x))
           
           # Layer 2  
           x = na.relu(self.fc2(x))
           
           # Layer 3
           x = na.relu(self.fc3(x))
           
           # Output layer with softmax
           x = na.softmax(self.output(x))
           
           return x
       
       def parameters(self):
           params = {}
           params.update(self.fc1.parameters())
           params.update(self.fc2.parameters())
           params.update(self.fc3.parameters())
           params.update(self.output.parameters())
           return params

Training with Batches
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create model
   model = MNISTClassifier()
   optimizer = na.Adam(model.parameters(), lr=0.001)
   
   # Training configuration
   epochs = 100
   batch_size = 32
   
   print("Starting MNIST training...")
   
   for epoch in range(epochs):
       epoch_loss = 0.0
       num_batches = 0
       
       # Simulate batch training
       for batch_idx in range(4):  # Simulate 4 batches
           # Get batch data (in practice, use DataLoader)
           batch_x, batch_y = prepare_mnist_data()
           
           # Forward pass
           predictions = model.forward(batch_x)
           
           # Compute cross-entropy loss
           loss = na.cross_entropy_loss(predictions, batch_y)
           
           # Backward pass
           loss.backward()
           optimizer.step()
           optimizer.zero_grad()
           
           epoch_loss += loss.data
           num_batches += 1
       
       # Print epoch results
       avg_loss = epoch_loss / num_batches
       if epoch % 10 == 0:
           print(f"Epoch {epoch:3d}, Average Loss: {avg_loss:.4f}")
   
   print("MNIST training completed!")

Part 4: Advanced Features
-------------------------

Configuration Management
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create training configuration
   config = na.Config({
       'model': {
           'type': 'mlp',
           'hidden_sizes': [512, 256, 128],
           'num_classes': 10,
           'activation': 'relu'
       },
       'training': {
           'learning_rate': 0.001,
           'batch_size': 32,
           'epochs': 100,
           'optimizer': 'adam'
       },
       'data': {
           'input_size': 784,
           'normalize': True
       }
   })
   
   # Save configuration
   na.save_config(config, 'mnist_config.yaml')
   
   # Load and use configuration
   loaded_config = na.load_config('mnist_config.yaml')
   lr = loaded_config.training.learning_rate
   print(f"Learning rate from config: {lr}")

Model Saving and Loading
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Save model parameters
   def save_model(model, filepath):
       """Save model parameters to file."""
       import pickle
       
       params = model.parameters()
       param_data = {name: param.data for name, param in params.items()}
       
       with open(filepath, 'wb') as f:
           pickle.dump(param_data, f)
       
       print(f"Model saved to {filepath}")
   
   def load_model(model, filepath):
       """Load model parameters from file."""
       import pickle
       
       with open(filepath, 'rb') as f:
           param_data = pickle.load(f)
       
       params = model.parameters()
       for name, param in params.items():
           if name in param_data:
               param.data = param_data[name]
       
       print(f"Model loaded from {filepath}")
   
   # Usage
   save_model(model, 'mnist_model.pkl')
   
   # Later, load the model
   new_model = MNISTClassifier()
   load_model(new_model, 'mnist_model.pkl')

Performance Monitoring
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import time
   
   def benchmark_model(model, input_data, num_runs=100):
       """Benchmark model inference speed."""
       
       # Warm up
       for _ in range(10):
           _ = model.forward(input_data)
       
       # Benchmark
       start_time = time.time()
       
       with na.no_grad():
           for _ in range(num_runs):
               predictions = model.forward(input_data)
       
       end_time = time.time()
       
       avg_time = (end_time - start_time) / num_runs
       throughput = len(input_data.data) / avg_time
       
       print(f"Average inference time: {avg_time*1000:.2f} ms")
       print(f"Throughput: {throughput:.0f} samples/second")
   
   # Benchmark the model
   test_data = na.Tensor(np.random.randn(32, 784))
   benchmark_model(model, test_data)

Part 5: Transformer Models
--------------------------

Building a Simple Transformer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class SimpleTransformer:
       def __init__(self, vocab_size, d_model, num_heads, num_layers):
           self.vocab_size = vocab_size
           self.d_model = d_model
           
           # Embedding layer
           self.embedding = na.Embedding(vocab_size, d_model)
           
           # Transformer blocks
           self.transformer_blocks = []
           for _ in range(num_layers):
               block = na.TransformerBlock(d_model, num_heads, d_model * 4)
               self.transformer_blocks.append(block)
           
           # Output projection
           self.output_proj = na.Linear(d_model, vocab_size)
       
       def forward(self, token_ids):
           # Embed tokens
           x = self.embedding(token_ids)
           
           # Apply transformer blocks
           for block in self.transformer_blocks:
               x = block(x)
           
           # Project to vocabulary
           output = na.softmax(self.output_proj(x))
           
           return output
       
       def parameters(self):
           params = {}
           params.update(self.embedding.parameters())
           
           for i, block in enumerate(self.transformer_blocks):
               block_params = block.parameters()
               for name, param in block_params.items():
                   params[f'transformer_{i}_{name}'] = param
           
           params.update(self.output_proj.parameters())
           return params

Text Generation Example
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create a small transformer for text generation
   vocab_size = 1000
   d_model = 256
   num_heads = 8
   num_layers = 4
   
   transformer = SimpleTransformer(vocab_size, d_model, num_heads, num_layers)
   optimizer = na.Adam(transformer.parameters(), lr=0.0001)
   
   # Simulate training data (sequences of token IDs)
   def generate_training_data(batch_size=16, seq_len=32):
       # Random token sequences
       sequences = np.random.randint(0, vocab_size, (batch_size, seq_len))
       return na.Tensor(sequences)
   
   # Training loop
   print("Training transformer...")
   for epoch in range(50):
       # Generate batch
       input_ids = generate_training_data()
       
       # Forward pass
       predictions = transformer.forward(input_ids)
       
       # Simple loss: predict next token
       targets = input_ids  # Self-supervised learning
       loss = na.cross_entropy_loss(predictions, targets)
       
       # Backward pass
       loss.backward()
       optimizer.step()
       optimizer.zero_grad()
       
       if epoch % 10 == 0:
           print(f"Epoch {epoch}, Loss: {loss.data:.4f}")

Next Steps
----------

Congratulations! You've completed the Neural Architecture tutorial. Here's what to explore next:

1. **Advanced Topics**: 
   - Custom gradient functions
   - Model parallelism
   - Memory optimization techniques

2. **Real Applications**:
   - Load real datasets (MNIST, CIFAR-10, text corpora)
   - Build production-ready training pipelines
   - Deploy models for inference

3. **Performance Optimization**:
   - Profile your models
   - Optimize memory usage
   - Use the CLI tools for automated training

4. **Community**:
   - Contribute to the project
   - Share your models and experiments
   - Help others in the community

Happy building with Neural Architecture! 🚀