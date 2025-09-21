# Demonstrate 'like' operations
print("'Like' operations:")
zeros_like_tensor = torch.zeros_like(tensor_2d)
print('torch.zeros_like(tensor_2d):')
print(zeros_like_tensor)
print()

ones_like_tensor = torch.ones_like(tensor_2d)
print('torch.ones_like(tensor_2d):')
print(ones_like_tensor)
print()

empty_like_tensor = torch.empty_like(tensor_2d)
print('torch.empty_like(tensor_2d):')
print(empty_like_tensor)
print()
import torch

print("PyTorch version:", torch.__version__)


# Set the random seed for reproducibility
torch.manual_seed(42)
print('torch.manual_seed(42) set for reproducibility')
print()

# Create an uninitialized tensor
tensor_empty = torch.empty(5, 2)
print('torch.empty(5, 2):')
print(tensor_empty)
print()

# Create a tensor filled with zeros
tensor_zeros = torch.zeros(5, 2)
print('torch.zeros(5, 2):')
print(tensor_zeros)
print()

# Create a tensor filled with ones
tensor_ones = torch.ones(5, 2)
print('torch.ones(5, 2):')
print(tensor_ones)
print()

torch.manual_seed(42)
# Create a tensor with random values from a uniform distribution on [0, 1)
tensor_rand = torch.rand(5, 2)
print('torch.rand(5, 2):')
print(tensor_rand)
print()

torch.manual_seed(42)
tensor_rand = torch.rand(5, 2)
print('torch.rand(5, 2):')
print(tensor_rand)
print()

# Create a tensor with random integers between 0 and 10
tensor_randint = torch.randint(0, 10, (5, 2))
print('torch.randint(0, 10, (5, 2))')
print(tensor_randint)
print()

# Create an identity matrix
tensor_eye = torch.eye(5)
print('torch.eye(5):')
print(tensor_eye)
print()

# Create a 1D tensor with values from 0 to 9
tensor_arange = torch.arange(10)
print('torch.arange(10):')
print(tensor_arange)
print()

# Create a 1D tensor with 5 values evenly spaced from 0 to 1
tensor_linspace = torch.linspace(0, 1, steps=5)
print('torch.linspace(0, 1, steps=5):')
print(tensor_linspace)
print()




# Create a 2D tensor
tensor_2d = torch.tensor([[1, 2, 3], [4, 5, 6]])
print('2D tensor:')
print(tensor_2d)
print('Shape:', tensor_2d.shape)
print('Dtype:', tensor_2d.dtype)
print('Type:', type(tensor_2d))
print()

# Do some operations
sum_tensor = tensor_2d.sum()
print('Sum of all elements:', sum_tensor.item())
print('Mean of all elements:', tensor_2d.float().mean().item())
print('Transpose:')
print(tensor_2d.t())
print()

# Convert integer tensor to float
tensor_2d_float = tensor_2d.float()
print('Converted to float:')
print(tensor_2d_float)
print('Dtype after conversion:', tensor_2d_float.dtype)