#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

__global__ void convolution_2D_basic(float* input, float* filter, float* output, int radius, int width, int height)
{
	int outCol = blockIdx.x * blockDim.x + threadIdx.x;
	int outRow = blockIdx.y * blockDim.y + threadIdx.y;

	float sum = 0.0f;

	for (int filterRow = 0; filterRow < 2 * radius + 1; filterRow++)
	{
		for (int filterCol = 0;filterCol < 2 * radius + 1;filterCol++)
		{
			int inRow = outRow - radius + filterRow;
			int inCol = outCol - radius + filterCol;
			if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width)
			{
				sum += filter[filterRow][filterCol] * input[inRow * width + inCol];
			}
		}
	}
	output[outRow][outCol] = sum;
}

// Wrapper function to call the kernel
torch::Tensor convolution_2D_basic_cuda(torch::Tensor input, torch::Tensor filter, torch::int radius) {
    // Input validation
    TORCH_CHECK(input.device().is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(filter.device().is_cuda(), "filter must be a CUDA tensor");
    

    // Create output tensor
    auto output = torch::empty_like(input);

    // Get pointers to the tensor data
    float* c_ptr = c.data_ptr<float>();
    const float* input_ptr = a.data_ptr<float>();
    const float* filter_ptr = b.data_ptr<float>();

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
    m.def("convolution_2D_basic", &convolution_2D_basic_cuda, "Convolution of two arrays (CUDA)");
}