#include<iostream>
#include<mma.h>
#include<cuda.h>
#include<cuda_runtime_api.h>
#include<device_launch_parameters.h>
#include <cuda_fp16.h>

using namespace nvcuda;

#define N 256
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void wmma_vector_add(const half* a, const half* b, float* c)
{
	wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>a_frag;
	wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>b_frag;
	wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

	wmma::load_matrix_sync(a_frag, a, WMMA_M);
	wmma::load_matrix_sync(b_frag, b, WMMA_M);

	wmma::fill_fragment(c_frag, 0.0f);

	for (int i = 0;i < c_frag.num_elements;i++)
	{
		c_frag.x[i] = __half2float(a_frag.x[i]) + __half2float(b_frag.x[i]);
	}

	wmma::store_matrix_sync(c, c_frag, WMMA_N, wmma::mem_row_major);

}

int main()
{
	half* h_a = new half[N * N];
	half* h_b = new half[N * N];
	float* h_c = new float[N * N];

	for (int i = 0;i < N * N;i++)
	{
		h_a[i] = __float2half(1.0f);
		h_b[i] = __float2half(2.0f);

	}

	half* d_a, *d_b;
	float* d_c;

	cudaMalloc(&d_a, N * N * sizeof(half));
	cudaMalloc(&d_b, N * N * sizeof(half));
	cudaMalloc(&d_c, N * N * sizeof(float));

	cudaMemcpy(d_a, h_a, N * N * sizeof(half), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, N * N * sizeof(half), cudaMemcpyHostToDevice);
	wmma_vector_add << <16, 32 >> > (d_a, d_b, d_c);
	cudaMemcpy(h_c, d_c, N * N * sizeof(float), cudaMemcpyDeviceToHost);

	std::cout << "Result: ";
	for (int i = 0; i < N; ++i) {
		std::cout << h_c[i] << " ";
		if (i % 10 == 0) 
			std::cout << std::endl;
	}
	std::cout << std::endl;

	delete[] h_a;
	delete[] h_b;
	delete[] h_c;
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	return 0;
}



