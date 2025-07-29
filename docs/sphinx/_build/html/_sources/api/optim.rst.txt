Optimization Module
==================

The optim module provides optimization algorithms for training neural networks.

.. currentmodule:: neural_arch.optim

Optimizers
----------

Adam Optimizer
~~~~~~~~~~~~~

.. autoclass:: Adam
   :members:
   :special-members: __init__
   :show-inheritance:

SGD Optimizer
~~~~~~~~~~~~

.. autoclass:: SGD
   :members:
   :special-members: __init__
   :show-inheritance:

AdamW Optimizer
~~~~~~~~~~~~~~

.. autoclass:: AdamW
   :members:
   :special-members: __init__
   :show-inheritance:

Examples
--------

Basic Usage
~~~~~~~~~~

.. code-block:: python

   import neural_arch as na
   
   # Create a simple model
   model = na.Linear(10, 1)
   
   # Setup optimizer
   optimizer = na.Adam(model.parameters(), lr=0.001)
   
   # Training loop
   for epoch in range(100):
       # Forward pass
       predictions = model(inputs)
       loss = na.mse_loss(predictions, targets)
       
       # Backward pass
       loss.backward()
       
       # Update parameters
       optimizer.step()
       
       # Clear gradients
       optimizer.zero_grad()

Adam Optimizer
~~~~~~~~~~~~~

Adam is the recommended optimizer for most neural network training:

.. code-block:: python

   # Adam with default parameters
   optimizer = na.Adam(model.parameters(), lr=0.001)
   
   # Adam with custom parameters
   optimizer = na.Adam(
       model.parameters(),
       lr=0.001,
       beta1=0.9,      # Momentum parameter
       beta2=0.999,    # RMSprop parameter
       eps=1e-8,       # Numerical stability
       weight_decay=0.01  # L2 regularization
   )

SGD Optimizer
~~~~~~~~~~~~

Stochastic Gradient Descent with optional momentum:

.. code-block:: python

   # Basic SGD
   optimizer = na.SGD(model.parameters(), lr=0.01)
   
   # SGD with momentum
   optimizer = na.SGD(
       model.parameters(),
       lr=0.01,
       momentum=0.9,
       weight_decay=1e-4
   )

AdamW Optimizer
~~~~~~~~~~~~~~

Adam with decoupled weight decay:

.. code-block:: python

   # AdamW for better generalization
   optimizer = na.AdamW(
       model.parameters(),
       lr=0.001,
       weight_decay=0.01  # Decoupled weight decay
   )

Learning Rate Scheduling
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Manual learning rate scheduling
   optimizer = na.Adam(model.parameters(), lr=0.001)
   
   for epoch in range(100):
       # Decay learning rate every 30 epochs
       if epoch % 30 == 0 and epoch > 0:
           optimizer.lr *= 0.1
       
       # Training step
       loss = train_step(model, optimizer, data)
       
       print(f"Epoch {epoch}, LR: {optimizer.lr}, Loss: {loss}")

Multiple Parameter Groups
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Different learning rates for different parts
   model = ComplexModel()
   
   # Get different parameter groups
   backbone_params = model.backbone.parameters()
   head_params = model.head.parameters()
   
   # Create optimizer with different learning rates
   # Note: Current implementation uses single parameter dict
   # This is a conceptual example for future enhancement
   all_params = {}
   all_params.update(backbone_params)
   all_params.update(head_params)
   
   optimizer = na.Adam(all_params, lr=0.001)

Advanced Training Patterns
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Gradient clipping example
   optimizer = na.Adam(model.parameters(), lr=0.001)
   
   for epoch in range(100):
       total_loss = 0
       for batch in dataloader:
           # Forward pass
           predictions = model(batch.inputs)
           loss = na.cross_entropy_loss(predictions, batch.targets)
           
           # Backward pass
           loss.backward()
           
           # Optional: Gradient clipping (handled automatically by tensors)
           # Gradients are clipped during backward pass for numerical stability
           
           # Update parameters
           optimizer.step()
           optimizer.zero_grad()
           
           total_loss += loss.data
       
       avg_loss = total_loss / len(dataloader)
       print(f"Epoch {epoch}, Average Loss: {avg_loss}")

Optimizer State
~~~~~~~~~~~~~~

.. code-block:: python

   # Access optimizer internal state
   optimizer = na.Adam(model.parameters(), lr=0.001)
   
   # Train for some steps
   for i in range(10):
       loss = train_step(model, optimizer, data)
   
   # Check optimizer state
   print(f"Step count: {optimizer.step_count}")
   print(f"Learning rate: {optimizer.lr}")
   
   # Adam-specific state
   if hasattr(optimizer, 'm'):
       print("Momentum estimates available")
   if hasattr(optimizer, 'v'):
       print("RMSprop estimates available")

Performance Tips
---------------

**Optimizer Choice**:
   - Use **Adam** for most applications (adaptive learning rates)
   - Use **SGD with momentum** for well-tuned hyperparameters
   - Use **AdamW** when regularization is important

**Learning Rate**:
   - Start with 0.001 for Adam
   - Start with 0.01-0.1 for SGD
   - Use learning rate scheduling for better convergence

**Batch Size**:
   - Larger batches work better with higher learning rates
   - Smaller batches may need more training steps

**Gradient Clipping**:
   - Automatically handled by the tensor system
   - Prevents gradient explosion in deep networks
   - Threshold set to 1e6 for numerical stability

Memory Considerations
-------------------

- **Parameter Storage**: Optimizers maintain references to model parameters
- **State Variables**: Adam stores momentum and RMSprop estimates (doubles memory)
- **Gradient Storage**: Gradients are stored in parameter tensors
- **Efficient Updates**: In-place parameter updates minimize memory allocation

Implementation Notes
------------------

The optimizers are implemented with:

- **Numerical Stability**: Careful handling of edge cases and numerical precision
- **Gradient Clipping**: Automatic clipping of extreme gradients
- **Error Handling**: Comprehensive validation of parameters and state
- **Performance**: Efficient NumPy operations for parameter updates
- **Flexibility**: Support for different parameter types and configurations