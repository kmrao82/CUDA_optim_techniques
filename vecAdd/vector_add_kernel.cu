#include <torch/extension.h>

__global__ void vector_add_kernel(const float* a, const float* b, float* c, int size)
{	
	
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < size)
	{
		c[idx] = a[idx] + b[idx];
	}
}

torch::Tensor vector_add_cuda(torch::Tensor a, torch::Tensor b)
{
	auto c = torch::empty_like(a);
	int size = a.numel();
	float milliseconds =0.0;
	int threads = 1024; 
	int blocks = (size +threads -1)/threads;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	vector_add_kernel<<<blocks,threads>>>(a.data_ptr<float>(),b.data_ptr<float>(),c.data_ptr<float>(),size);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Kernel execution time: %.3f milliseconds\n", milliseconds);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	return c; 
}