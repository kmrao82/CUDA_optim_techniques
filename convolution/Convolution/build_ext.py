from torch import conv2d
from torch.utils.cpp_extension import load

# This builds the extension in-place
custom_cuda = load(
    name="custom_cuda_kernels",
    sources=["kernel.cu"],
    verbose=True,
    is_python_module=True,
    with_cuda=True
)

# Test the extension here if you want
if __name__ == "__main__":
    import torch 
    
    # Create test tensors on GPU
    a = torch.rand(1000, device="cuda").float()
    f = torch.rand(5, device="cuda").float()
    
    # Call your custom kernel
    c_custom = custom_cuda.convolution_2D_basic(a, f)
    
    # Check result against PyTorch's native implementation
    c_torch = conv2d(a,f, device="cuda")
    
    # Verify results
    max_diff = (c_custom - c_torch).abs().max().item()
    print(f"Maximum difference: {max_diff}")
    assert max_diff < 1e-6, "Results don't match!"
    
    print("Test passed!")