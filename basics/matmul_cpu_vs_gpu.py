import torch
import time

# Create two 5000x5000 tensors
A = torch.rand(5000, 5000)
B = torch.rand(5000, 5000)

# Matrix multiplication on CPU
t0 = time.time()
C_cpu = torch.matmul(A, B)
cpu_time = time.time() - t0
print(f"CPU time: {cpu_time:.6f} seconds")

# Check if CUDA is available for GPU
if torch.cuda.is_available():
    A_gpu = A.to('cuda')
    B_gpu = B.to('cuda')
    torch.cuda.synchronize()  # Ensure all previous CUDA ops are done
    t0 = time.time()
    C_gpu = torch.matmul(A_gpu, B_gpu)
    torch.cuda.synchronize()  # Wait for the operation to finish
    gpu_time = time.time() - t0
    print(f"GPU time: {gpu_time:.6f} seconds")
else:
    print("CUDA (GPU) is not available on this system.")
