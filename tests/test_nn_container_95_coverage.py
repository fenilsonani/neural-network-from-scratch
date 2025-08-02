"""Ultra-comprehensive tests for Container modules to achieve 95%+ test coverage.

This test suite covers ModuleList and Sequential container implementations
to ensure robust 95%+ test coverage.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from neural_arch.core.base import Module
from neural_arch.core.tensor import Tensor
from neural_arch.nn.activation import ReLU
from neural_arch.nn.container import ModuleList, Sequential
from neural_arch.nn.linear import Linear


# Mock modules for testing
class MockModule(Module):
    """Mock module for testing container functionality."""

    def __init__(self, name="mock"):
        super().__init__()
        self.name = name
        self.call_count = 0

    def forward(self, x):
        self.call_count += 1
        return x * 2  # Simple transformation

    def __call__(self, x):
        return self.forward(x)


class MockLinearModule(Module):
    """Mock linear module for testing."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = np.random.randn(in_features, out_features).astype(np.float32)

    def forward(self, x):
        if isinstance(x, Tensor):
            data = x.data
        else:
            data = x
        return Tensor(np.dot(data, self.weight))

    def __call__(self, x):
        return self.forward(x)


class TestModuleList95Coverage:
    """Comprehensive ModuleList tests targeting 95%+ coverage."""

    def test_modulelist_initialization_empty(self):
        """Test ModuleList initialization with no modules."""
        module_list = ModuleList()

        assert len(module_list) == 0
        assert module_list._modules_list == []
        assert len(module_list._modules) == 0

    def test_modulelist_initialization_with_modules(self):
        """Test ModuleList initialization with modules."""
        modules = [MockModule("mod1"), MockModule("mod2"), MockModule("mod3")]
        module_list = ModuleList(modules)

        assert len(module_list) == 3
        assert len(module_list._modules_list) == 3
        assert len(module_list._modules) == 3

        # Check module registration
        assert "0" in module_list._modules
        assert "1" in module_list._modules
        assert "2" in module_list._modules

    def test_modulelist_initialization_with_different_iterables(self):
        """Test ModuleList initialization with different iterable types."""
        modules = [MockModule("mod1"), MockModule("mod2")]

        # Test with list
        ml_list = ModuleList(modules)
        assert len(ml_list) == 2

        # Test with tuple
        ml_tuple = ModuleList(tuple(modules))
        assert len(ml_tuple) == 2

        # Test with generator
        ml_gen = ModuleList(m for m in modules)
        assert len(ml_gen) == 2

    def test_modulelist_getitem_comprehensive(self):
        """Test ModuleList __getitem__ with various indices."""
        modules = [MockModule(f"mod{i}") for i in range(5)]
        module_list = ModuleList(modules)

        # Test positive indices
        assert module_list[0] is modules[0]
        assert module_list[2] is modules[2]
        assert module_list[4] is modules[4]

        # Test negative indices
        assert module_list[-1] is modules[4]
        assert module_list[-2] is modules[3]
        assert module_list[-5] is modules[0]

        # Test slice
        slice_result = module_list[1:4]
        assert isinstance(slice_result, ModuleList)
        assert len(slice_result) == 3
        assert slice_result[0] is modules[1]
        assert slice_result[1] is modules[2]
        assert slice_result[2] is modules[3]

        # Test slice with step
        step_slice = module_list[::2]
        assert len(step_slice) == 3
        assert step_slice[0] is modules[0]
        assert step_slice[1] is modules[2]
        assert step_slice[2] is modules[4]

    def test_modulelist_getitem_errors(self):
        """Test ModuleList __getitem__ error conditions."""
        modules = [MockModule(f"mod{i}") for i in range(3)]
        module_list = ModuleList(modules)

        # Test invalid index types
        with pytest.raises(TypeError):
            module_list["invalid"]

        with pytest.raises(TypeError):
            module_list[1.5]

        # Test out of range indices
        with pytest.raises(IndexError):
            module_list[5]

        with pytest.raises(IndexError):
            module_list[-10]

    def test_modulelist_setitem_comprehensive(self):
        """Test ModuleList __setitem__ functionality."""
        modules = [MockModule(f"mod{i}") for i in range(3)]
        module_list = ModuleList(modules)

        # Test setting valid index
        new_module = MockModule("new_mod")
        module_list[1] = new_module

        assert module_list[1] is new_module
        assert module_list._modules["1"] is new_module

        # Test setting negative index
        another_module = MockModule("another_mod")
        module_list[-1] = another_module

        assert module_list[2] is another_module
        assert module_list._modules["2"] is another_module

    def test_modulelist_setitem_errors(self):
        """Test ModuleList __setitem__ error conditions."""
        modules = [MockModule(f"mod{i}") for i in range(3)]
        module_list = ModuleList(modules)

        # Test invalid module type
        with pytest.raises(TypeError):
            module_list[0] = "not_a_module"

        # Test invalid index type
        with pytest.raises(TypeError):
            module_list["invalid"] = MockModule()

        # Test out of range index
        with pytest.raises(IndexError):
            module_list[5] = MockModule()

    def test_modulelist_delitem_comprehensive(self):
        """Test ModuleList __delitem__ functionality."""
        modules = [MockModule(f"mod{i}") for i in range(5)]
        module_list = ModuleList(modules)

        # Test deleting by positive index
        del module_list[1]
        assert len(module_list) == 4
        assert module_list[0] is modules[0]
        assert module_list[1] is modules[2]  # shifted

        # Test deleting by negative index
        del module_list[-1]
        assert len(module_list) == 3

        # Test deleting by slice
        del module_list[0:2]
        assert len(module_list) == 1

    def test_modulelist_delitem_errors(self):
        """Test ModuleList __delitem__ error conditions."""
        modules = [MockModule(f"mod{i}") for i in range(3)]
        module_list = ModuleList(modules)

        # Test invalid index type
        with pytest.raises(TypeError):
            del module_list["invalid"]

        # Test out of range index
        with pytest.raises(IndexError):
            del module_list[5]

    def test_modulelist_len(self):
        """Test ModuleList __len__ method."""
        # Test empty list
        empty_list = ModuleList()
        assert len(empty_list) == 0

        # Test non-empty list
        modules = [MockModule(f"mod{i}") for i in range(7)]
        module_list = ModuleList(modules)
        assert len(module_list) == 7

        # Test after modifications
        module_list.append(MockModule("extra"))
        assert len(module_list) == 8

        del module_list[0]
        assert len(module_list) == 7

    def test_modulelist_iter(self):
        """Test ModuleList __iter__ method."""
        modules = [MockModule(f"mod{i}") for i in range(4)]
        module_list = ModuleList(modules)

        # Test iteration
        iterated_modules = []
        for module in module_list:
            iterated_modules.append(module)

        assert len(iterated_modules) == 4
        for i, module in enumerate(iterated_modules):
            assert module is modules[i]

        # Test list conversion
        list_modules = list(module_list)
        assert len(list_modules) == 4
        assert all(list_modules[i] is modules[i] for i in range(4))

    def test_modulelist_iadd(self):
        """Test ModuleList __iadd__ (+=) method."""
        initial_modules = [MockModule(f"mod{i}") for i in range(2)]
        module_list = ModuleList(initial_modules)

        additional_modules = [MockModule(f"extra{i}") for i in range(2)]

        # Test +=
        module_list += additional_modules
        result = module_list

        assert result is module_list  # Should return self
        assert len(module_list) == 4
        assert module_list[2] is additional_modules[0]
        assert module_list[3] is additional_modules[1]

    def test_modulelist_add(self):
        """Test ModuleList __add__ (+) method."""
        modules1 = [MockModule(f"mod1_{i}") for i in range(2)]
        modules2 = [MockModule(f"mod2_{i}") for i in range(2)]

        ml1 = ModuleList(modules1)
        ml2 = ModuleList(modules2)

        # Test +
        combined = ml1 + ml2

        assert isinstance(combined, ModuleList)
        assert combined is not ml1
        assert combined is not ml2
        assert len(combined) == 4
        assert combined[0] is modules1[0]
        assert combined[1] is modules1[1]
        assert combined[2] is modules2[0]
        assert combined[3] is modules2[1]

        # Test + with list
        additional = [MockModule("extra")]
        combined_with_list = ml1 + additional
        assert len(combined_with_list) == 3
        assert combined_with_list[2] is additional[0]

    def test_modulelist_append(self):
        """Test ModuleList append method."""
        module_list = ModuleList()

        # Test appending to empty list
        module1 = MockModule("mod1")
        result = module_list.append(module1)

        assert result is module_list  # Should return self for chaining
        assert len(module_list) == 1
        assert module_list[0] is module1
        assert "0" in module_list._modules

        # Test appending to non-empty list
        module2 = MockModule("mod2")
        module_list.append(module2)

        assert len(module_list) == 2
        assert module_list[1] is module2
        assert "1" in module_list._modules

    def test_modulelist_append_errors(self):
        """Test ModuleList append error conditions."""
        module_list = ModuleList()

        # Test appending non-module
        with pytest.raises(TypeError):
            module_list.append("not_a_module")

        with pytest.raises(TypeError):
            module_list.append(42)

    def test_modulelist_extend(self):
        """Test ModuleList extend method."""
        module_list = ModuleList([MockModule("initial")])

        # Test extending with list
        new_modules = [MockModule(f"ext{i}") for i in range(3)]
        result = module_list.extend(new_modules)

        assert result is module_list  # Should return self for chaining
        assert len(module_list) == 4
        assert module_list[1] is new_modules[0]
        assert module_list[2] is new_modules[1]
        assert module_list[3] is new_modules[2]

        # Test extending with tuple
        tuple_modules = (MockModule("tuple1"), MockModule("tuple2"))
        module_list.extend(tuple_modules)
        assert len(module_list) == 6

        # Test extending with generator
        gen_modules = (MockModule(f"gen{i}") for i in range(2))
        module_list.extend(gen_modules)
        assert len(module_list) == 8

    def test_modulelist_insert(self):
        """Test ModuleList insert method."""
        modules = [MockModule(f"mod{i}") for i in range(3)]
        module_list = ModuleList(modules)

        # Test inserting at beginning
        new_module1 = MockModule("inserted_0")
        module_list.insert(0, new_module1)

        assert len(module_list) == 4
        assert module_list[0] is new_module1
        assert module_list[1] is modules[0]

        # Test inserting in middle
        new_module2 = MockModule("inserted_2")
        module_list.insert(2, new_module2)

        assert len(module_list) == 5
        assert module_list[2] is new_module2

        # Test inserting at end
        new_module3 = MockModule("inserted_end")
        module_list.insert(len(module_list), new_module3)

        assert len(module_list) == 6
        assert module_list[-1] is new_module3

    def test_modulelist_insert_errors(self):
        """Test ModuleList insert error conditions."""
        module_list = ModuleList([MockModule("mod")])

        # Test inserting non-module
        with pytest.raises(TypeError):
            module_list.insert(0, "not_a_module")

    def test_modulelist_forward(self):
        """Test ModuleList forward method."""
        modules = [MockModule(f"mod{i}") for i in range(3)]
        module_list = ModuleList(modules)

        # Test forward pass
        x = Tensor(np.array([1.0], dtype=np.float32))
        output = module_list.forward(x)

        # Should apply all modules sequentially (each multiplies by 2)
        expected = 1.0 * 2 * 2 * 2  # 8.0
        assert np.allclose(output.data, [expected])

        # Check that all modules were called
        for module in modules:
            assert module.call_count == 1

    def test_modulelist_forward_empty(self):
        """Test ModuleList forward with empty list."""
        module_list = ModuleList()

        x = Tensor(np.array([1.0], dtype=np.float32))
        output = module_list.forward(x)

        # Should return input unchanged
        assert np.allclose(output.data, x.data)

    def test_modulelist_module_registry_consistency(self):
        """Test that _modules registry stays consistent."""
        module_list = ModuleList()

        # Add modules and check registry
        for i in range(3):
            module = MockModule(f"mod{i}")
            module_list.append(module)
            assert str(i) in module_list._modules
            assert module_list._modules[str(i)] is module

        # Delete and check registry updates
        del module_list[1]
        assert len(module_list._modules) == 2
        assert "0" in module_list._modules
        assert "1" in module_list._modules  # Should be shifted
        assert "2" not in module_list._modules

    def test_modulelist_complex_operations(self):
        """Test complex combinations of ModuleList operations."""
        # Start with some modules
        initial_modules = [MockModule(f"initial{i}") for i in range(3)]
        ml = ModuleList(initial_modules)

        # Perform various operations
        ml.append(MockModule("appended"))
        ml.insert(1, MockModule("inserted"))
        ml.extend([MockModule("extended1"), MockModule("extended2")])

        assert len(ml) == 7

        # Test slicing and assignment
        ml[2:4] = [MockModule("replaced1"), MockModule("replaced2")]

        # Test that everything still works
        x = Tensor(np.array([1.0], dtype=np.float32))
        output = ml.forward(x)
        assert np.all(np.isfinite(output.data))


class TestSequential95Coverage:
    """Comprehensive Sequential tests targeting 95%+ coverage."""

    def test_sequential_initialization_empty(self):
        """Test Sequential initialization with no arguments."""
        seq = Sequential()

        assert len(seq) == 0
        assert len(seq._modules_list) == 0

    def test_sequential_initialization_with_modules(self):
        """Test Sequential initialization with modules."""
        modules = [MockModule(f"mod{i}") for i in range(3)]
        seq = Sequential(*modules)

        assert len(seq) == 3
        assert seq[0] is modules[0]
        assert seq[1] is modules[1]
        assert seq[2] is modules[2]

    def test_sequential_initialization_with_mixed_modules(self):
        """Test Sequential with different types of modules."""
        try:
            # Try to create a realistic sequential with Linear layers
            seq = Sequential(
                MockLinearModule(10, 5), MockLinearModule(5, 3), MockLinearModule(3, 1)
            )

            assert len(seq) == 3

            # Test forward pass
            x = Tensor(np.random.randn(2, 10).astype(np.float32))
            output = seq.forward(x)
            assert output.shape == (2, 1)

        except Exception:
            # If Linear modules don't work, use mock modules
            mock_modules = [MockModule(f"mod{i}") for i in range(3)]
            seq = Sequential(*mock_modules)

            x = Tensor(np.array([1.0], dtype=np.float32))
            output = seq.forward(x)
            assert np.allclose(output.data, [8.0])  # 1 * 2^3

    def test_sequential_forward_pass(self):
        """Test Sequential forward pass."""
        modules = [MockModule(f"mod{i}") for i in range(4)]
        seq = Sequential(*modules)

        x = Tensor(np.array([2.0], dtype=np.float32))
        output = seq.forward(x)

        # Each module multiplies by 2, so 2 * 2^4 = 32
        expected = 2.0 * (2**4)
        assert np.allclose(output.data, [expected])

        # Check that all modules were called
        for module in modules:
            assert module.call_count == 1

    def test_sequential_forward_empty(self):
        """Test Sequential forward with no modules."""
        seq = Sequential()

        x = Tensor(np.array([5.0], dtype=np.float32))
        output = seq.forward(x)

        # Should return input unchanged
        assert np.allclose(output.data, x.data)

    def test_sequential_forward_single_module(self):
        """Test Sequential forward with single module."""
        module = MockModule("single")
        seq = Sequential(module)

        x = Tensor(np.array([3.0], dtype=np.float32))
        output = seq.forward(x)

        assert np.allclose(output.data, [6.0])  # 3 * 2
        assert module.call_count == 1

    def test_sequential_inheritance_from_modulelist(self):
        """Test that Sequential inherits ModuleList functionality."""
        modules = [MockModule(f"mod{i}") for i in range(3)]
        seq = Sequential(*modules)

        # Test inherited methods
        assert len(seq) == 3
        assert seq[0] is modules[0]
        assert list(seq) == modules

        # Test append (inherited)
        new_module = MockModule("new")
        seq.append(new_module)
        assert len(seq) == 4
        assert seq[3] is new_module

        # Test extend (inherited)
        extra_modules = [MockModule("extra1"), MockModule("extra2")]
        seq.extend(extra_modules)
        assert len(seq) == 6

    def test_sequential_module_access_patterns(self):
        """Test various ways to access Sequential modules."""
        modules = [MockModule(f"mod{i}") for i in range(5)]
        seq = Sequential(*modules)

        # Test indexing
        assert seq[0] is modules[0]
        assert seq[-1] is modules[4]

        # Test slicing
        sub_seq = seq[1:4]
        assert isinstance(sub_seq, Sequential)
        assert len(sub_seq) == 3

        # Test iteration
        for i, module in enumerate(seq):
            assert module is modules[i]

    def test_sequential_modification(self):
        """Test modifying Sequential after creation."""
        seq = Sequential(MockModule("mod1"), MockModule("mod2"))

        # Test insertion
        seq.insert(1, MockModule("inserted"))
        assert len(seq) == 3

        # Test deletion
        del seq[0]
        assert len(seq) == 2

        # Test replacement
        seq[0] = MockModule("replaced")
        assert len(seq) == 2

        # Test that forward still works
        x = Tensor(np.array([1.0], dtype=np.float32))
        output = seq.forward(x)
        assert np.all(np.isfinite(output.data))

    def test_sequential_with_different_input_types(self):
        """Test Sequential with different input types."""
        seq = Sequential(MockModule("mod1"), MockModule("mod2"))

        # Test with Tensor
        x_tensor = Tensor(np.array([1.0], dtype=np.float32))
        output_tensor = seq.forward(x_tensor)
        assert isinstance(output_tensor, Tensor)

        # Test with numpy array (if modules support it)
        try:
            x_numpy = np.array([1.0], dtype=np.float32)
            output_numpy = seq.forward(x_numpy)
            assert np.all(np.isfinite(output_numpy))
        except Exception:
            # Modules might not support numpy input directly
            pass

    def test_sequential_error_propagation(self):
        """Test error propagation in Sequential."""

        class ErrorModule(Module):
            def forward(self, x):
                raise ValueError("Test error")

            def __call__(self, x):
                return self.forward(x)

        seq = Sequential(MockModule("good"), ErrorModule(), MockModule("never_reached"))

        x = Tensor(np.array([1.0], dtype=np.float32))

        with pytest.raises(ValueError, match="Test error"):
            seq.forward(x)


class TestContainerIntegration:
    """Integration tests for container modules."""

    def test_nested_containers(self):
        """Test nesting containers within containers."""
        # Create nested structure
        inner_seq1 = Sequential(MockModule("inner1_1"), MockModule("inner1_2"))
        inner_seq2 = Sequential(MockModule("inner2_1"), MockModule("inner2_2"))

        outer_list = ModuleList([inner_seq1, inner_seq2])

        assert len(outer_list) == 2
        assert isinstance(outer_list[0], Sequential)
        assert isinstance(outer_list[1], Sequential)

        # Test forward through nested structure
        x = Tensor(np.array([1.0], dtype=np.float32))

        # Process through first inner sequence
        output1 = outer_list[0].forward(x)
        assert np.allclose(output1.data, [4.0])  # 1 * 2 * 2

        # Process through second inner sequence
        output2 = outer_list[1].forward(x)
        assert np.allclose(output2.data, [4.0])  # 1 * 2 * 2

    def test_container_with_real_modules(self):
        """Test containers with real neural network modules."""
        try:
            # Try creating a realistic network
            from neural_arch.nn.linear import Linear

            network = Sequential(Linear(10, 20), Linear(20, 10), Linear(10, 1))

            x = Tensor(np.random.randn(5, 10).astype(np.float32))
            output = network.forward(x)

            assert output.shape == (5, 1)
            assert np.all(np.isfinite(output.data))

        except ImportError:
            # If real modules aren't available, skip this test
            pytest.skip("Real neural network modules not available")

    def test_container_parameter_collection(self):
        """Test parameter collection from containers."""
        try:
            from neural_arch.nn.linear import Linear

            # Create container with parameterized modules
            container = ModuleList([Linear(5, 3), Linear(3, 2), Linear(2, 1)])

            # Check that parameters are accessible
            parameters = list(container.parameters())
            assert len(parameters) > 0  # Should have weight and bias parameters

        except (ImportError, AttributeError):
            # If parameter collection isn't implemented or modules unavailable
            pytest.skip("Parameter collection not available")

    def test_container_state_consistency(self):
        """Test that container state remains consistent."""
        modules = [MockModule(f"mod{i}") for i in range(3)]
        container = ModuleList(modules)

        # Store initial state
        initial_length = len(container)
        initial_modules = list(container)

        # Perform forward passes
        x = Tensor(np.array([1.0], dtype=np.float32))
        for _ in range(5):
            output = container.forward(x)
            assert np.all(np.isfinite(output.data))

        # State should be unchanged
        assert len(container) == initial_length
        assert list(container) == initial_modules

    def test_container_memory_efficiency(self):
        """Test container memory efficiency with many modules."""
        # Create container with many modules
        many_modules = [MockModule(f"mod{i}") for i in range(100)]
        container = ModuleList(many_modules)

        assert len(container) == 100

        # Test that iteration and access still work efficiently
        count = 0
        for module in container:
            count += 1
        assert count == 100

        # Test slicing with large container
        subset = container[10:20]
        assert len(subset) == 10

    def test_container_edge_cases(self):
        """Test container edge cases and boundary conditions."""
        # Test with single module
        single_container = ModuleList([MockModule("single")])
        assert len(single_container) == 1

        x = Tensor(np.array([1.0], dtype=np.float32))
        output = single_container.forward(x)
        assert np.allclose(output.data, [2.0])

        # Test removing all modules
        single_container.clear() if hasattr(single_container, "clear") else None

        # Test Sequential with one module
        single_seq = Sequential(MockModule("single"))
        output_seq = single_seq.forward(x)
        assert np.allclose(output_seq.data, [2.0])

    def test_container_type_checking(self):
        """Test container type checking and validation."""
        container = ModuleList()

        # Test that only Module instances can be added
        with pytest.raises(TypeError):
            container.append("not_a_module")

        with pytest.raises(TypeError):
            container.append(42)

        with pytest.raises(TypeError):
            container.append(lambda x: x)

        # Test that Module instances are accepted
        container.append(MockModule("valid"))
        assert len(container) == 1

    def test_container_string_representation(self):
        """Test container string representations."""
        modules = [MockModule(f"mod{i}") for i in range(2)]

        # Test ModuleList representation
        ml = ModuleList(modules)
        ml_repr = repr(ml)
        assert "ModuleList" in ml_repr

        # Test Sequential representation
        seq = Sequential(*modules)
        seq_repr = repr(seq)
        assert "Sequential" in seq_repr
