Core Module
===========

The core module provides the fundamental building blocks of Neural Architecture, including the tensor system, automatic differentiation, and base classes.

.. currentmodule:: neural_arch.core

Tensor System
-------------

.. autoclass:: Tensor
   :members:
   :special-members: __init__, __add__, __mul__, __matmul__
   :show-inheritance:

Parameter System
---------------

.. autoclass:: Parameter
   :members:
   :special-members: __init__
   :show-inheritance:

Base Classes
-----------

.. autoclass:: Module
   :members:
   :special-members: __init__, __call__
   :show-inheritance:

Device and Data Types
--------------------

.. autoclass:: Device
   :members:
   :show-inheritance:

.. autoclass:: DType
   :members:
   :show-inheritance:

Utility Functions
----------------

.. autofunction:: get_default_device
.. autofunction:: set_default_device
.. autofunction:: get_default_dtype
.. autofunction:: set_default_dtype

Gradient Control
---------------

.. autofunction:: no_grad
.. autofunction:: enable_grad
.. autofunction:: is_grad_enabled

Context Managers
---------------

The core module provides context managers for controlling gradient computation:

.. code-block:: python

   import neural_arch as na
   
   # Disable gradients for inference
   with na.no_grad():
       predictions = model(inputs)
   
   # Explicitly enable gradients
   with na.enable_grad():
       loss = compute_loss(predictions, targets)
       loss.backward()

Examples
--------

Basic Tensor Operations
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import neural_arch as na
   
   # Create tensors
   x = na.Tensor([[1, 2], [3, 4]], requires_grad=True)
   y = na.Tensor([[2, 1], [0, 1]], requires_grad=True)
   
   # Operations
   z = na.matmul(x, y)
   loss = na.mean_pool(z)
   
   # Compute gradients
   loss.backward()
   print(f"x.grad: {x.grad}")
   print(f"y.grad: {y.grad}")

Parameter Management
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create a parameter
   weight = na.Parameter(na.Tensor([[0.1, 0.2], [0.3, 0.4]]))
   
   # Use in computation
   output = na.matmul(input_tensor, weight)
   
   # Parameters track gradients automatically
   loss.backward()
   print(f"Weight gradients: {weight.grad}")

Device Management
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Check current device
   current_device = na.get_default_device()
   print(f"Default device: {current_device}")
   
   # Set device (CPU only in current implementation)
   na.set_default_device(na.Device.CPU)