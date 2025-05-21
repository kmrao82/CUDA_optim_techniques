#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <c10/cuda/CUDAStream.h>

#define TILE_WIDTH 768

// A simple CUDA kernel
//__global__ void add_arrays_kernel(float* c, const float* a, const float* b, int size) {

    //1. Implementation with Gmem<->smem and addition
    //int bx = blockIdx.x;
    //int tx = threadIdx.x;
    //__shared__ float a_smem[TILE_WIDTH];
    //__shared__ float b_smem[TILE_WIDTH];

    //int idx = blockIdx.x * blockDim.x + tx;
    //if (idx < size) {
    //    
    //    a_smem[tx] = a[idx];
    //    b_smem[tx] = b[idx];
    //    //Add loop unrolling here using float4 ?

    //}
    //__syncthreads();

    //
    //if (idx < size)
    //{
    //    c[idx] = a_smem[tx]+b_smem[tx];
    //}
    // }
    //2. Implementation with float4 or constant memory.
 __global__ void add_arrays_cuda(float4 * __restrict__ c, const float4 * __restrict__ a,
        const float4 * __restrict__ b, int num_float4){
     const int idx = blockIdx.x * blockDim.x + threadIdx.x;
     if (idx < num_float4)
     {
         const float4 a_val = a[idx];
         const float4 b_val = b[idx];
         c[idx] = make_float4(
             a_val.x + b_val.x,
             a_val.y + b_val.y,
             a_val.z + b_val.z,
             a_val.w + b_val.w
             );
     }
}

 __global__ void add_arrays_scalar(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int offset, int remainder)
 {
     const int idx = blockIdx.x * blockDim.x + threadIdx.x;
     const int element_idx = offset + idx;
     if (idx < remainder && element_idx <(offset+remainder))
     {
         c[element_idx] = a[element_idx] + b[element_idx];
     }
}
// Wrapper function to call the kernel
// 1. For implementation with Gmem <-> smem
// 
//torch::Tensor add_arrays_cuda(torch::Tensor a, torch::Tensor b) {
//    // Input validation
//    TORCH_CHECK(a.device().is_cuda(), "a must be a CUDA tensor");
//    TORCH_CHECK(b.device().is_cuda(), "b must be a CUDA tensor");
//    TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have the same shape");
//
//    // Create output tensor
//    auto c = torch::empty_like(a);
//
//    // Get pointers to the tensor data
//    float* c_ptr = c.data_ptr<float>();
//    const float* a_ptr = a.data_ptr<float>();
//    const float* b_ptr = b.data_ptr<float>();
//
//    // Calculate grid and block dimensions
//    int size = a.numel();
//    //int threads = 256;
//    //int blocks = (size + threads - 1) / threads;
//    int blockSize;
//    int minGridSize;
//    int gridSize;
//
//    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, add_arrays_kernel, 0, 0);
//    gridSize = (size + blockSize - 1) / blockSize;
//
//    // Launch kernel
//    add_arrays_kernel << <gridSize, blockSize >> > (c_ptr, a_ptr, b_ptr, size);
//    cudaDeviceSynchronize();
//
//    //Calculate the theoretical Occupancy;
//    int device;
//    cudaDeviceProp props;
//    cudaGetDevice(&device);
//    cudaGetDeviceProperties(&props, device);
//
//    int maxActiveBlocks;
//    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
//        &maxActiveBlocks,
//        add_arrays_kernel,
//        blockSize,
//        0
//    );
//
//    float occupancy = (maxActiveBlocks * blockSize / props.warpSize)
//        / (float)(props.maxThreadsPerMultiProcessor / props.warpSize);
//
//    printf("Block size: %d, Theoretical occupancy: %.2f\n", blockSize, occupancy);
//
//    // Check for errors
//    cudaError_t err = cudaGetLastError();
//    if (err != cudaSuccess) {
//        TORCH_CHECK(false, "CUDA kernel error: ", cudaGetErrorString(err));
//    }
//
//    return c;
//}

 //Wrapper function to call the kernel 
 //2. Implementation with float4 
 torch::Tensor add_arrays_cuda_wrapper(torch::Tensor a, torch::Tensor b)
 {
     TORCH_CHECK(a.device().is_cuda(), "a must be contiguous");
     TORCH_CHECK(b.device().is_cuda(), "b must be contiguous");
     TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have same shape");

     auto c = torch::empty_like(a);

     const float* a_ptr = a.data_ptr<float>();
     const float* b_ptr = b.data_ptr<float>();
     float* c_ptr = c.data_ptr<float>();

     const int size = a.numel();
     const int num_float4 = size / 4;
     const int remainder = size % 4;

    //int threads = 256;
    //int blocks = (size + threads - 1) / threads;
    int blockSize;
    int minGridSize;
    int gridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, add_arrays_cuda, 0, 0);
  
    if (num_float4 > 0)
    {
        gridSize = (num_float4 + blockSize - 1) / blockSize;
        const float4* a4 = reinterpret_cast<const float4*>(a_ptr);
        const float4* b4 = reinterpret_cast<const float4*>(b_ptr);
        float4* results = reinterpret_cast<float4*>(c_ptr);
        add_arrays_cuda << <gridSize, blockSize, 0, at::cuda::getCurrentCUDAStream() >> > (results, a4, b4, num_float4);
    }

    if (remainder > 0)
    {   
        const int offset = num_float4 * 4;
        add_arrays_scalar << < 1, remainder, 0, at::cuda::getCurrentCUDAStream() >> > (a_ptr, b_ptr, c_ptr, offset, remainder);
    }
//     Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel error: ", cudaGetErrorString(err));
    }

    return c;
 }

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_arrays", &add_arrays_cuda_wrapper, "Add two arrays (CUDA)");
  
}