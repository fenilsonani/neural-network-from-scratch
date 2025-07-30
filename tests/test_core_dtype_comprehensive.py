"""Comprehensive tests for core/dtype to improve coverage from 78.85% to 100%.

This file tests DType enum and dtype management functions.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np
from neural_arch.core.dtype import DType, get_default_dtype, set_default_dtype


class TestDTypeComprehensive:
    """Comprehensive tests for DType enum."""
    
    def test_dtype_enum_values(self):
        """Test all DType enum values."""
        assert DType.FLOAT32.value == np.float32
        assert DType.FLOAT64.value == np.float64
        assert DType.INT32.value == np.int32
        assert DType.INT64.value == np.int64
        assert DType.BOOL.value == np.bool_
        
        # Test that all values are numpy types
        for dtype in DType:
            assert isinstance(dtype.value, type)
            assert issubclass(dtype.value, (np.generic, bool))
    
    def test_numpy_dtype_property(self):
        """Test numpy_dtype property."""
        assert DType.FLOAT32.numpy_dtype == np.float32
        assert DType.FLOAT64.numpy_dtype == np.float64
        assert DType.INT32.numpy_dtype == np.int32
        assert DType.INT64.numpy_dtype == np.int64
        assert DType.BOOL.numpy_dtype == np.bool_
    
    def test_is_floating_property(self):
        """Test is_floating property."""
        assert DType.FLOAT32.is_floating is True
        assert DType.FLOAT64.is_floating is True
        assert DType.INT32.is_floating is False
        assert DType.INT64.is_floating is False
        assert DType.BOOL.is_floating is False
    
    def test_is_integer_property(self):
        """Test is_integer property."""
        assert DType.FLOAT32.is_integer is False
        assert DType.FLOAT64.is_integer is False
        assert DType.INT32.is_integer is True
        assert DType.INT64.is_integer is True
        assert DType.BOOL.is_integer is False
    
    def test_bytes_per_element_property(self):
        """Test bytes_per_element property."""
        assert DType.FLOAT32.bytes_per_element == 4
        assert DType.FLOAT64.bytes_per_element == 8
        assert DType.INT32.bytes_per_element == 4
        assert DType.INT64.bytes_per_element == 8
        assert DType.BOOL.bytes_per_element == 1
    
    def test_from_numpy_with_numpy_dtype(self):
        """Test from_numpy with np.dtype objects."""
        # Test with np.dtype objects
        assert DType.from_numpy(np.dtype(np.float32)) == DType.FLOAT32
        assert DType.from_numpy(np.dtype(np.float64)) == DType.FLOAT64
        assert DType.from_numpy(np.dtype(np.int32)) == DType.INT32
        assert DType.from_numpy(np.dtype(np.int64)) == DType.INT64
        assert DType.from_numpy(np.dtype(np.bool_)) == DType.BOOL
    
    def test_from_numpy_with_type(self):
        """Test from_numpy with numpy type classes."""
        # Test with type objects
        assert DType.from_numpy(np.float32) == DType.FLOAT32
        assert DType.from_numpy(np.float64) == DType.FLOAT64
        assert DType.from_numpy(np.int32) == DType.INT32
        assert DType.from_numpy(np.int64) == DType.INT64
        assert DType.from_numpy(np.bool_) == DType.BOOL
    
    def test_from_numpy_with_string(self):
        """Test from_numpy with string dtype names."""
        # Test with string names
        assert DType.from_numpy('float32') == DType.FLOAT32
        assert DType.from_numpy('float64') == DType.FLOAT64
        assert DType.from_numpy('int32') == DType.INT32
        assert DType.from_numpy('int64') == DType.INT64
        assert DType.from_numpy('bool') == DType.BOOL
        
        # Alternative string names
        assert DType.from_numpy('f4') == DType.FLOAT32
        assert DType.from_numpy('f8') == DType.FLOAT64
        assert DType.from_numpy('i4') == DType.INT32
        assert DType.from_numpy('i8') == DType.INT64
    
    def test_from_numpy_unsupported(self):
        """Test from_numpy with unsupported dtypes."""
        # Unsupported dtypes should raise ValueError
        with pytest.raises(ValueError, match="Unsupported NumPy dtype"):
            DType.from_numpy(np.complex64)
        
        with pytest.raises(ValueError, match="Unsupported NumPy dtype"):
            DType.from_numpy('complex128')
        
        with pytest.raises(ValueError, match="Unsupported NumPy dtype"):
            DType.from_numpy(np.dtype('U10'))  # Unicode string
        
        # Test with float16 (not in our enum)
        with pytest.raises(ValueError, match="Unsupported NumPy dtype"):
            DType.from_numpy(np.float16)
    
    def test_str_method(self):
        """Test __str__ method."""
        assert str(DType.FLOAT32) == "float32"
        assert str(DType.FLOAT64) == "float64"
        assert str(DType.INT32) == "int32"
        assert str(DType.INT64) == "int64"
        assert str(DType.BOOL) == "bool"
    
    def test_repr_method(self):
        """Test __repr__ method."""
        assert repr(DType.FLOAT32) == "DType.FLOAT32"
        assert repr(DType.FLOAT64) == "DType.FLOAT64"
        assert repr(DType.INT32) == "DType.INT32"
        assert repr(DType.INT64) == "DType.INT64"
        assert repr(DType.BOOL) == "DType.BOOL"
    
    def test_dtype_comparison(self):
        """Test DType comparison and identity."""
        # Same dtype should be equal
        assert DType.FLOAT32 == DType.FLOAT32
        assert DType.INT64 == DType.INT64
        
        # Different dtypes should not be equal
        assert DType.FLOAT32 != DType.FLOAT64
        assert DType.INT32 != DType.INT64
        assert DType.BOOL != DType.FLOAT32
        
        # Identity checks
        assert DType.FLOAT32 is DType.FLOAT32
        assert DType.INT64 is DType.INT64
    
    def test_dtype_hash(self):
        """Test DType can be used in sets and dicts."""
        dtype_set = {DType.FLOAT32, DType.INT32, DType.FLOAT32}
        assert len(dtype_set) == 2  # Duplicates removed
        
        dtype_dict = {
            DType.FLOAT32: "float",
            DType.INT32: "int",
            DType.BOOL: "bool"
        }
        assert dtype_dict[DType.FLOAT32] == "float"
        assert dtype_dict[DType.INT32] == "int"
        assert dtype_dict[DType.BOOL] == "bool"
    
    def test_dtype_iteration(self):
        """Test iterating over DType enum."""
        all_dtypes = list(DType)
        assert len(all_dtypes) == 5
        assert DType.FLOAT32 in all_dtypes
        assert DType.FLOAT64 in all_dtypes
        assert DType.INT32 in all_dtypes
        assert DType.INT64 in all_dtypes
        assert DType.BOOL in all_dtypes


class TestDTypeManagement:
    """Tests for dtype management functions."""
    
    def test_get_default_dtype(self):
        """Test get_default_dtype function."""
        # Default should be FLOAT32
        assert get_default_dtype() == DType.FLOAT32
        
        # Should always return a DType
        assert isinstance(get_default_dtype(), DType)
    
    def test_set_default_dtype(self):
        """Test set_default_dtype function."""
        # Save original default
        original = get_default_dtype()
        
        try:
            # Test setting different dtypes
            set_default_dtype(DType.FLOAT64)
            assert get_default_dtype() == DType.FLOAT64
            
            set_default_dtype(DType.INT32)
            assert get_default_dtype() == DType.INT32
            
            set_default_dtype(DType.BOOL)
            assert get_default_dtype() == DType.BOOL
            
            # Test all dtypes
            for dtype in DType:
                set_default_dtype(dtype)
                assert get_default_dtype() == dtype
        
        finally:
            # Restore original default
            set_default_dtype(original)
    
    def test_set_default_dtype_invalid(self):
        """Test set_default_dtype with invalid input."""
        # Save original default
        original = get_default_dtype()
        
        try:
            # Should raise TypeError for non-DType values
            with pytest.raises(TypeError, match="Expected DType"):
                set_default_dtype(np.float32)
            
            with pytest.raises(TypeError, match="Expected DType"):
                set_default_dtype("float32")
            
            with pytest.raises(TypeError, match="Expected DType"):
                set_default_dtype(None)
            
            with pytest.raises(TypeError, match="Expected DType"):
                set_default_dtype(32)
            
            # Default should not have changed
            assert get_default_dtype() == original
        
        finally:
            # Ensure default is restored
            set_default_dtype(original)
    
    def test_dtype_numpy_conversion_roundtrip(self):
        """Test conversion between DType and numpy dtypes."""
        for dtype in DType:
            # Convert to numpy and back
            numpy_dtype = dtype.numpy_dtype
            recovered = DType.from_numpy(numpy_dtype)
            assert recovered == dtype
            
            # Also test with np.dtype wrapper
            numpy_dtype_obj = np.dtype(numpy_dtype)
            recovered2 = DType.from_numpy(numpy_dtype_obj)
            assert recovered2 == dtype
    
    def test_dtype_array_creation(self):
        """Test creating numpy arrays with DType."""
        # Create arrays with each dtype
        data = [1, 2, 3, 4, 5]
        
        for dtype in DType:
            arr = np.array(data, dtype=dtype.numpy_dtype)
            assert arr.dtype == dtype.numpy_dtype
            
            # Verify bytes per element
            assert arr.itemsize == dtype.bytes_per_element
    
    def test_dtype_type_aliases(self):
        """Test type aliases are defined."""
        from neural_arch.core.dtype import FloatDType, IntDType
        
        # These are just type aliases, ensure they exist
        assert FloatDType is not None
        assert IntDType is not None