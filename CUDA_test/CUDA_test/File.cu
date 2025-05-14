#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>


//CUDA Kernel for vec Addition
__global__ void vecAdd(float* a, float* b, float* c, int num)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num)
	{
		c[idx] = a[idx] + b[idx];
	}
}


int main()
{
	int N = 1024;
	float* a, * b, * c;

	for (int i = 0;i < N;i++)
	{
		a[i] = 1.0f;
		b[i] = 2.0f;
	}
	float* a_d, * b_d, * c_d;
	cudaMalloc(&a_d, N * sizeof(float));
	cudaMalloc(&b_d, N * sizeof(float));
	cudaMalloc(&c_d, N * sizeof(float));

	cudaMemcpy(a_d, a, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(b_d, b, N * sizeof(float), cudaMemcpyHostToDevice);

	int blockSize = 256;
	int gridSize = (N + blockSize - 1) / blockSize;

	vecAdd << <gridSize, blockSize >> > (a_d, b_d, c_d, N);

	cudaMemcpy(a, a_d, N * sizeof(float), cudaMemcpyDeviceToHost);

	for (int i = 0;i < N;i++)
		std::cout << c[i] << " " << std::endl;

	cudaFree(a_d);
	cudaFree(b_d);
	cudaFree(c_d);

	delete a;
	delete b;
	delete c;

	return 0;

}