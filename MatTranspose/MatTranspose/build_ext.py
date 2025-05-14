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
    a = torch.randn(100,100, device="cuda").float()
    b = torch.empty_like(a);

    # Call your custom kernel
    c_custom = custom_cuda.mat_transpose(a, b)
    print(c_custom)

    # Check result against PyTorch's native implementation
    c_torch = a.transpose(0,1);
    print(c_torch)

    # Verify results
    max_diff = (c_custom - c_torch).abs().max().item()
    print(f"Maximum difference: {max_diff}")
    assert max_diff < 1e-6, "Results don't match!"
    
    print("Test passed!")