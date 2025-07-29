Configuration Module
===================

The config module provides enterprise-grade configuration management for Neural Architecture applications.

.. currentmodule:: neural_arch.config

Configuration Classes
--------------------

.. autoclass:: Config
   :members:
   :special-members: __init__
   :show-inheritance:

Configuration Functions
----------------------

.. autofunction:: load_config
.. autofunction:: save_config
.. autofunction:: get_preset_config

Examples
--------

Basic Configuration Usage
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import neural_arch as na
   
   # Load configuration from file
   config = na.load_config('config.yaml')
   
   # Access configuration values
   learning_rate = config.training.learning_rate
   batch_size = config.training.batch_size
   model_type = config.model.type
   
   # Use in model creation
   if model_type == 'transformer':
       model = create_transformer_model(config.model)
   elif model_type == 'mlp':
       model = create_mlp_model(config.model)

Creating Configurations
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create configuration programmatically
   config = na.Config({
       'model': {
           'type': 'transformer',
           'd_model': 512,
           'num_heads': 8,
           'num_layers': 6
       },
       'training': {
           'learning_rate': 0.001,
           'batch_size': 32,
           'epochs': 100,
           'optimizer': 'adam'
       },
       'data': {
           'train_path': 'data/train.txt',
           'val_path': 'data/val.txt',
           'vocab_size': 10000
       }
   })
   
   # Save configuration
   na.save_config(config, 'my_config.yaml')

Preset Configurations
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Get preset configurations for common scenarios
   
   # Small transformer for experimentation
   small_config = na.get_preset_config('transformer_small')
   
   # Large model for production
   large_config = na.get_preset_config('transformer_large')
   
   # Simple MLP configuration
   mlp_config = na.get_preset_config('mlp_basic')
   
   # Print available presets
   presets = na.get_preset_config('list')
   print("Available presets:", presets)

Configuration File Formats
~~~~~~~~~~~~~~~~~~~~~~~~~

YAML format (recommended):

.. code-block:: yaml

   # config.yaml
   model:
     type: transformer
     d_model: 512
     num_heads: 8
     num_layers: 6
     dropout: 0.1
   
   training:
     learning_rate: 0.001
     batch_size: 32
     epochs: 100
     optimizer: adam
     weight_decay: 0.01
   
   data:
     train_path: data/train.txt
     val_path: data/val.txt
     vocab_size: 10000
     max_length: 512

JSON format:

.. code-block:: json

   {
     "model": {
       "type": "mlp",
       "hidden_sizes": [512, 256, 128],
       "activation": "relu",
       "dropout": 0.2
     },
     "training": {
       "learning_rate": 0.001,
       "batch_size": 64,
       "epochs": 50
     }
   }

Validation and Type Checking
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Configuration with validation
   config_schema = {
       'model.d_model': {'type': int, 'min': 64, 'max': 2048},
       'model.num_heads': {'type': int, 'min': 1, 'max': 32},
       'training.learning_rate': {'type': float, 'min': 1e-6, 'max': 1.0},
       'training.batch_size': {'type': int, 'min': 1, 'max': 1024}
   }
   
   # Load and validate configuration
   config = na.load_config('config.yaml', schema=config_schema)
   
   # Configuration will raise ValidationError if invalid
   try:
       config = na.Config({'training': {'learning_rate': 10.0}})  # Too high
   except na.ConfigValidationError as e:
       print(f"Invalid configuration: {e}")

Environment Variable Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Configuration can use environment variables
   config = na.Config({
       'training': {
           'learning_rate': '${LEARNING_RATE:0.001}',  # Default to 0.001
           'batch_size': '${BATCH_SIZE:32}',
           'device': '${DEVICE:cpu}'
       },
       'data': {
           'data_path': '${DATA_PATH}',  # Required environment variable
       }
   })
   
   # Environment variables are resolved when accessed
   lr = config.training.learning_rate  # Uses env var or default

Configuration Inheritance
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Base configuration
   base_config = na.Config({
       'model': {'type': 'transformer', 'd_model': 512},
       'training': {'epochs': 100, 'learning_rate': 0.001}
   })
   
   # Specialized configuration inheriting from base  
   experiment_config = base_config.copy()
   experiment_config.update({
       'training': {'learning_rate': 0.0005},  # Override learning rate
       'experiment': {'name': 'low_lr_test'}   # Add new section
   })
   
   # Save specialized configuration
   na.save_config(experiment_config, 'experiment.yaml')

Integration with Training
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Complete training setup with configuration
   config = na.load_config('training_config.yaml')
   
   # Create model from config
   if config.model.type == 'transformer':
       model = na.TransformerBlock(
           d_model=config.model.d_model,
           num_heads=config.model.num_heads,
           d_ff=config.model.d_ff
       )
   
   # Create optimizer from config
   if config.training.optimizer == 'adam':
       optimizer = na.Adam(
           model.parameters(),
           lr=config.training.learning_rate,
           weight_decay=config.training.get('weight_decay', 0.0)
       )
   
   # Training loop using config
   for epoch in range(config.training.epochs):
       # Load data using config paths
       train_data = load_data(config.data.train_path)
       
       # Train using config parameters
       train_epoch(model, optimizer, train_data, config)

Configuration Management Best Practices
-------------------------------------

**Organization**:
   - Group related settings into sections (model, training, data)
   - Use consistent naming conventions
   - Document configuration options

**Validation**:
   - Define schemas for critical parameters
   - Validate ranges and types
   - Provide helpful error messages

**Defaults**:
   - Provide sensible defaults for optional parameters
   - Use preset configurations for common scenarios
   - Document default values

**Environment Integration**:
   - Use environment variables for deployment-specific settings
   - Support both development and production configurations
   - Keep sensitive information in environment variables

**Version Control**:
   - Track configuration files in version control
   - Use different configs for different experiments
   - Document configuration changes

Preset Configurations Available
------------------------------

The module includes several preset configurations:

**Transformer Presets**:
   - ``transformer_small``: For experimentation and development
   - ``transformer_base``: Standard transformer configuration
   - ``transformer_large``: For production and large-scale training

**MLP Presets**:
   - ``mlp_basic``: Simple feedforward network
   - ``mlp_deep``: Deep feedforward network
   - ``mlp_wide``: Wide feedforward network

**Training Presets**:
   - ``training_fast``: Quick training for experimentation
   - ``training_standard``: Balanced training configuration
   - ``training_thorough``: Comprehensive training setup