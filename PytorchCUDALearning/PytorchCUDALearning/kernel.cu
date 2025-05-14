#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

// A simple CUDA kernel
__global__ void add_arrays_kernel(float* c, const float* a, const float* b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}

// Wrapper function to call the kernel
torch::Tensor add_arrays_cuda(torch::Tensor a, torch::Tensor b) {
    // Input validation
    TORCH_CHECK(a.device().is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.device().is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have the same shape");

    // Create output tensor
    auto c = torch::empty_like(a);

    // Get pointers to the tensor data
    float* c_ptr = c.data_ptr<float>();
    const float* a_ptr = a.data_ptr<float>();
    const float* b_ptr = b.data_ptr<float>();

    // Calculate grid and block dimensions
    int size = a.numel();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    // Launch kernel
    add_arrays_kernel << <blocks, threads >> > (c_ptr, a_ptr, b_ptr, size);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel error: ", cudaGetErrorString(err));
    }

    return c;
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_arrays", &add_arrays_cuda, "Add two arrays (CUDA)");
}