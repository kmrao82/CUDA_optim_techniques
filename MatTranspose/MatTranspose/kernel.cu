#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

__global__ void mat_transpose(const float* input, float* output, int width, int height)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (row < height && col < width)
	{
		output[col * height + row] = input[row * width + col];
	}
}


torch::Tensor mat_transpose_cuda(torch::Tensor a, torch::Tensor b)
{
	TORCH_CHECK(a.device().is_cuda(), " Input tensor must be a CUDA tensor");
	TORCH_CHECK(b.device().is_cuda(), " Input tensor must be a CUDA tensor");

	const float* a_ptr = a.data_ptr<float>();
	float* b_ptr = b.data_ptr<float>();
	
	
	int width = a.size(1);
	int height = a.size(0);

	/*dim3 blocksize =(16,16);
	dim3 gridsize = ((width + blocksize.x - 1) / blocksize.x , (height + blocksize.y-1)/blocksize.y) ;*/
	int threads = 1024;
	int blocks = (width + threads - 1) / threads;

	mat_transpose << <blocks,threads >> > (a_ptr, b_ptr, width, height);
	// Check for errors
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		TORCH_CHECK(false, "CUDA kernel error: ", cudaGetErrorString(err));
	}

	return b;
}




// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("mat_transpose", &mat_transpose_cuda, "Matrix transpose (CUDA)");
}
