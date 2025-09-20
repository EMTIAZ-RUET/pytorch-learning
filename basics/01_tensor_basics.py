"""
Basic Tensor Operations in PyTorch
==================================

This script demonstrates fundamental tensor operations in PyTorch.
Perfect starting point for learning PyTorch basics.
"""

import torch
import numpy as np

def tensor_creation():
    """Examples of creating tensors in different ways."""
    print("=== Tensor Creation ===")
    
    # From lists
    data = [[1, 2], [3, 4]]
    x_data = torch.tensor(data)
    print(f"From list: {x_data}")
    
    # From NumPy arrays
    np_array = np.array(data)
    x_np = torch.from_numpy(np_array)
    print(f"From NumPy: {x_np}")
    
    # With specific shapes and values
    x_ones = torch.ones(2, 3)
    print(f"Ones tensor: {x_ones}")
    
    x_rand = torch.rand(2, 3)
    print(f"Random tensor: {x_rand}")
    
    # With specific dtype and device
    x_float = torch.zeros(2, 3, dtype=torch.float32)
    print(f"Float tensor: {x_float}")
    

def tensor_operations():
    """Basic tensor operations."""
    print("\n=== Tensor Operations ===")
    
    # Create sample tensors
    x = torch.rand(2, 3)
    y = torch.rand(2, 3)
    
    print(f"Tensor x: {x}")
    print(f"Tensor y: {y}")
    
    # Element-wise operations
    z1 = x + y
    z2 = torch.add(x, y)
    print(f"Addition: {z1}")
    
    # Matrix multiplication
    x_matrix = torch.rand(2, 3)
    y_matrix = torch.rand(3, 2)
    z_matrix = torch.matmul(x_matrix, y_matrix)
    print(f"Matrix multiplication shape: {z_matrix.shape}")
    
    # In-place operations (end with _)
    x.add_(y)
    print(f"In-place addition: {x}")


def tensor_properties():
    """Exploring tensor properties."""
    print("\n=== Tensor Properties ===")
    
    tensor = torch.rand(3, 4)
    print(f"Tensor: {tensor}")
    print(f"Shape: {tensor.shape}")
    print(f"Data type: {tensor.dtype}")
    print(f"Device: {tensor.device}")
    print(f"Number of dimensions: {tensor.ndim}")
    print(f"Total elements: {tensor.numel()}")


def tensor_indexing():
    """Tensor indexing and slicing."""
    print("\n=== Tensor Indexing ===")
    
    tensor = torch.rand(4, 4)
    print(f"Original tensor: {tensor}")
    
    # First row
    print(f"First row: {tensor[0]}")
    
    # First column
    print(f"First column: {tensor[:, 0]}")
    
    # Last column
    print(f"Last column: {tensor[..., -1]}")
    
    # Specific element
    print(f"Element at [1,1]: {tensor[1, 1]}")


if __name__ == "__main__":
    print("PyTorch Tensor Basics Tutorial")
    print("=" * 50)
    
    tensor_creation()
    tensor_operations()
    tensor_properties()
    tensor_indexing()
    
    print("\n" + "=" * 50)
    print("Tutorial completed! Ready to move to the next topic.")