"""Test to understand parameter bug."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from neural_arch.nn import Linear
from neural_arch.core import Module, Parameter, Tensor
import numpy as np

class SimpleModule(Module):
    def __init__(self):
        super().__init__()
        self.linear = Linear(10, 5)
    
    def forward(self, x):
        return self.linear(x)

# Test direct parameter access
print("Testing parameter access...")
module = SimpleModule()

print("\n1. Direct _parameters access:")
print(f"module._parameters: {list(module._parameters.keys())}")
print(f"module.linear._parameters: {list(module.linear._parameters.keys())}")

print("\n2. parameters() method:")
params = module.parameters()
print(f"Type of params: {type(params)}")
print(f"params.keys(): {list(params.keys())}")

print("\n3. Iterating over parameters():")
for i, p in enumerate(module.parameters()):
    print(f"  Item {i}: type={type(p)}, value={p if isinstance(p, str) else p.data.shape}")

print("\n4. params.values():")
for i, p in enumerate(params.values()):
    print(f"  Value {i}: type={type(p)}, shape={p.data.shape if hasattr(p, 'data') else 'N/A'}")

print("\n5. params._get_params():")
actual_params = params._get_params()
for k, v in actual_params.items():
    print(f"  {k}: type={type(v)}, shape={v.data.shape if hasattr(v, 'data') else 'N/A'}")

# Test what the optimizer sees
from neural_arch.optim import Adam
print("\n6. What optimizer sees:")
optimizer = Adam(module.parameters(), lr=0.01)
print(f"Number of param groups: {len(optimizer.param_groups)}")
if optimizer.param_groups:
    print(f"Params in first group: {len(optimizer.param_groups[0]['params'])}")
    for i, p in enumerate(optimizer.param_groups[0]['params']):
        print(f"  Param {i}: type={type(p)}, value={p if isinstance(p, str) else 'tensor'}")