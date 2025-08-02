Command Line Interface
=====================

The CLI module provides command-line tools for training, evaluation, and configuration management.

.. currentmodule:: neural_arch.cli

CLI Functions
------------

.. autofunction:: main

Commands
--------

Train Command
~~~~~~~~~~~~

Train a neural network model:

.. code-block:: bash

   # Train with configuration file
   neural-arch train --config config.yaml --output models/my_model
   
   # Train with command line parameters
   neural-arch train --model-type transformer --lr 0.001 --epochs 100

Evaluate Command
~~~~~~~~~~~~~~~

Evaluate a trained model:

.. code-block:: bash

   # Evaluate model
   neural-arch eval --model models/my_model --data test_data.txt
   
   # Evaluate with custom metrics
   neural-arch eval --model models/my_model --data test_data.txt --metrics accuracy,f1

Config Command
~~~~~~~~~~~~~

Manage configurations:

.. code-block:: bash

   # Generate default configuration
   neural-arch config --generate --output config.yaml
   
   # Validate configuration
   neural-arch config --validate config.yaml
   
   # List available presets
   neural-arch config --list-presets

Benchmark Command
~~~~~~~~~~~~~~~~

Run performance benchmarks:

.. code-block:: bash

   # Benchmark tensor operations
   neural-arch benchmark --operations tensor
   
   # Benchmark training performance
   neural-arch benchmark --training --model transformer --batch-size 32

Examples
--------

Complete Training Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # 1. Generate configuration
   neural-arch config --generate --preset transformer_base --output my_config.yaml
   
   # 2. Edit configuration as needed
   # Edit my_config.yaml
   
   # 3. Validate configuration
   neural-arch config --validate my_config.yaml
   
   # 4. Train model
   neural-arch train --config my_config.yaml --output models/transformer_model
   
   # 5. Evaluate model
   neural-arch eval --model models/transformer_model --data test.txt

Programmatic CLI Usage
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import neural_arch as na
   
   # Run CLI commands programmatically
   exit_code = na.run_cli('train', '--config', 'config.yaml')
   
   # Handle different commands
   commands = [
       ['config', '--generate', '--output', 'config.yaml'],
       ['train', '--config', 'config.yaml'],
       ['eval', '--model', 'model.pth', '--data', 'test.txt']
   ]
   
   for cmd in commands:
       result = na.run_cli(*cmd)
       if result != 0:
           print(f"Command failed: {cmd}")

Configuration File Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Generate different types of configurations
   
   # Transformer configuration
   neural-arch config --generate --preset transformer_base --output transformer.yaml
   
   # MLP configuration
   neural-arch config --generate --preset mlp_basic --output mlp.yaml
   
   # Custom configuration with overrides
   neural-arch config --generate --preset transformer_base \
     --set model.d_model=768 \
     --set training.learning_rate=0.0005 \
     --output custom.yaml

Training with Different Options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Basic training
   neural-arch train --data train.txt --output model.pth
   
   # Training with validation
   neural-arch train --data train.txt --val-data val.txt --output model.pth
   
   # Training with specific architecture
   neural-arch train --model-type transformer \
     --d-model 512 --num-heads 8 --num-layers 6 \
     --data train.txt --output transformer_model.pth
   
   # Training with custom optimizer settings
   neural-arch train --data train.txt \
     --optimizer adam --lr 0.001 --weight-decay 0.01 \
     --batch-size 64 --epochs 100 \
     --output optimized_model.pth

Model Evaluation
~~~~~~~~~~~~~~~

.. code-block:: bash

   # Basic evaluation
   neural-arch eval --model model.pth --data test.txt
   
   # Evaluation with detailed metrics
   neural-arch eval --model model.pth --data test.txt \
     --metrics accuracy,precision,recall,f1 \
     --output results.json
   
   # Batch evaluation on multiple test sets
   neural-arch eval --model model.pth \
     --data test1.txt,test2.txt,test3.txt \
     --output eval_results/

Performance Benchmarking
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Benchmark core operations
   neural-arch benchmark --operations tensor,matrix,activation
   
   # Benchmark training performance
   neural-arch benchmark --training \
     --model transformer --batch-sizes 16,32,64,128 \
     --sequence-lengths 128,256,512
   
   # Memory usage benchmarks
   neural-arch benchmark --memory \
     --model transformer --max-batch-size 128
   
   # Save benchmark results
   neural-arch benchmark --all --output benchmarks.json

CLI Configuration
---------------

The CLI can be configured through:

**Configuration Files**:
   - Global config: ``~/.neural_arch/config.yaml``
   - Project config: ``./neural_arch.yaml``
   - Explicit config: ``--config path/to/config.yaml``

**Environment Variables**:
   - ``NEURAL_ARCH_CONFIG``: Path to default configuration
   - ``NEURAL_ARCH_LOG_LEVEL``: Logging level (DEBUG, INFO, WARNING, ERROR)
   - ``NEURAL_ARCH_DEVICE``: Default device (cpu)

**Command Line Arguments**:
   - Override any configuration value
   - Support for nested configuration keys
   - Type conversion and validation

Error Handling
-------------

The CLI provides comprehensive error handling:

.. code-block:: bash

   # Configuration validation errors
   neural-arch train --config invalid_config.yaml
   # Error: Invalid configuration: model.d_model must be > 0
   
   # File not found errors
   neural-arch train --data nonexistent.txt
   # Error: Training data file not found: nonexistent.txt
   
   # Parameter validation errors
   neural-arch train --lr -0.1
   # Error: Learning rate must be positive, got -0.1

Logging and Output
----------------

**Logging Levels**:
   - ``--verbose``: Enable debug logging
   - ``--quiet``: Suppress info messages
   - ``--log-file``: Write logs to file

**Progress Reporting**:
   - Training progress bars
   - Real-time loss reporting
   - ETA calculations

**Output Formats**:
   - JSON for structured data
   - YAML for configurations
   - Plain text for human-readable output

Integration Examples
------------------

**Shell Scripts**:

.. code-block:: bash

   #!/bin/bash
   # train_model.sh
   
   CONFIG_FILE="configs/experiment_1.yaml"
   MODEL_DIR="models/experiment_1"
   
   # Generate configuration if it doesn't exist
   if [ ! -f "$CONFIG_FILE" ]; then
       neural-arch config --generate --preset transformer_base --output "$CONFIG_FILE"
   fi
   
   # Train model
   neural-arch train --config "$CONFIG_FILE" --output "$MODEL_DIR"
   
   # Evaluate model
   neural-arch eval --model "$MODEL_DIR" --data test.txt

**Python Integration**:

.. code-block:: python

   import subprocess
   import neural_arch as na
   
   def run_training_pipeline(config_path, output_dir):
       """Run complete training pipeline via CLI."""
       
       # Validate configuration
       result = na.run_cli('config', '--validate', config_path)
       if result != 0:
           raise ValueError(f"Invalid configuration: {config_path}")
       
       # Train model
       result = na.run_cli('train', '--config', config_path, '--output', output_dir)
       if result != 0:
           raise RuntimeError("Training failed")
       
       # Evaluate model
       result = na.run_cli('eval', '--model', output_dir, '--data', 'test.txt')
       return result == 0

CLI Architecture
--------------

The CLI is built with:

- **Argument Parsing**: Uses argparse for robust command-line parsing
- **Configuration Management**: Seamless integration with config system
- **Error Handling**: Comprehensive error reporting and validation
- **Extensibility**: Plugin architecture for custom commands
- **Testing**: Full test coverage for all CLI functionality

The CLI serves as the primary interface for production deployments and automated training pipelines.