Functional Module
=================

The functional module provides low-level tensor operations that form the building blocks of neural networks.

.. currentmodule:: neural_arch.functional

Arithmetic Operations
--------------------

.. autofunction:: add
.. autofunction:: sub
.. autofunction:: mul
.. autofunction:: div
.. autofunction:: neg
.. autofunction:: matmul

Activation Functions
-------------------

.. autofunction:: relu
.. autofunction:: sigmoid  
.. autofunction:: tanh
.. autofunction:: softmax

Pooling Operations
-----------------

.. autofunction:: mean_pool
.. autofunction:: max_pool

Loss Functions
-------------

.. autofunction:: cross_entropy_loss
.. autofunction:: mse_loss

Utility Functions
----------------

.. automodule:: neural_arch.functional.utils
   :members:

Examples
--------

Basic Operations
~~~~~~~~~~~~~~~

.. code-block:: python

   import neural_arch.functional as F
   import neural_arch as na
   
   # Create tensors
   x = na.Tensor([[1, 2], [3, 4]], requires_grad=True)
   y = na.Tensor([[2, 1], [0, 1]], requires_grad=True)
   
   # Arithmetic operations
   z1 = F.add(x, y)        # Element-wise addition
   z2 = F.mul(x, y)        # Element-wise multiplication
   z3 = F.matmul(x, y)     # Matrix multiplication
   
   # Activation functions
   activated = F.relu(z3)
   probabilities = F.softmax(activated)

Broadcasting Examples
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Broadcasting in arithmetic operations
   x = na.Tensor([[1, 2, 3]], requires_grad=True)  # Shape: (1, 3)
   y = na.Tensor([[1], [2], [3]], requires_grad=True)  # Shape: (3, 1)
   
   # Broadcasts to (3, 3)
   result = F.add(x, y)
   print(f"Result shape: {result.shape}")  # (3, 3)

Activation Function Usage
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # ReLU activation
   x = na.Tensor([-1, 0, 1, 2], requires_grad=True)
   relu_output = F.relu(x)  # [0, 0, 1, 2]
   
   # Softmax for probability distributions
   logits = na.Tensor([[1, 2, 3], [2, 1, 3]], requires_grad=True)
   probs = F.softmax(logits)  # Each row sums to 1
   
   # Sigmoid for binary classification
   binary_logits = na.Tensor([0.5, -0.5, 2.0], requires_grad=True)
   binary_probs = F.sigmoid(binary_logits)

Loss Function Examples
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Cross-entropy loss for classification
   predictions = na.Tensor([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]], requires_grad=True)
   targets = na.Tensor([0, 1])  # Class indices
   ce_loss = F.cross_entropy_loss(predictions, targets)
   
   # Mean squared error for regression
   predicted_values = na.Tensor([1.0, 2.0, 3.0], requires_grad=True)
   actual_values = na.Tensor([1.1, 1.9, 3.2])
   mse_loss = F.mse_loss(predicted_values, actual_values)

Performance Notes
----------------

The functional operations are implemented with NumPy for optimal performance:

- **Vectorized Operations**: All operations use NumPy's vectorized implementations
- **Broadcasting**: Automatic broadcasting following NumPy conventions
- **Memory Efficiency**: Operations minimize memory allocations where possible
- **BLAS Integration**: Matrix operations leverage optimized BLAS libraries

The functional API is designed to be composable and efficient, serving as the foundation for higher-level neural network modules.