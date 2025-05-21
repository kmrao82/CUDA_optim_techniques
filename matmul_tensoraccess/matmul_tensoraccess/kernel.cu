#include<iostream>
#include<cuda.h>
#include<cstdlib>
#include<mma.h>
#include<cuda_fp16.h>
#include<device_launch_parameters.h>

using namespace nvcuda;

//WMMA matrix dimensions (16 x 16 x 16 for fp16)
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

//Matrix dimensions 
#define M 16
#define N 16
#define K 16

__global__ void wmma_matmul(half* a, half* b, float* c)
{
	// Declare Shared memory
	__shared__ half ashared[WMMA_M][WMMA_K];
	__shared__ half bshared[WMMA_K][WMMA_N];

	// Declare WMMA fragments
	wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>a_frag;
	wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>b_frag;
	wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

	// Initialize accumulator to zeros;
	wmma::fill_fragment(acc_frag, 0.0f);

	//Tile indices
	int tileRow = blockIdx.y * WMMA_M + threadIdx.y;
	int tileCol = blockIdx.x * WMMA_N + threadIdx.x;                                                                                

	//Load tiles from global memory to shared memory
	for (int i = 0;i < K;i += WMMA_K)
	{
		ashared[threadIdx.y][threadIdx.x] = a[(tileRow) * K + (i + threadIdx.x)];
		bshared[threadIdx.y][threadIdx.x] = b[(i + threadIdx.y) * N + (tileCol)];
		__syncthreads();

		//Load fragments from shared memory
		wmma::load_matrix_sync(a_frag, &ashared[0][0], WMMA_K);
		wmma::load_matrix_sync(b_frag, &bshared[0][0],  WMMA_N);
		
		//Perform matrix multiplication
		wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
		__syncthreads();
	}

	//Store results to Global memory
	wmma::store_matrix_sync(&c[tileRow * N + tileCol], acc_frag, N, wmma::mem_row_major);

}






int main()
{
	half* h_a = new half[M * K];
	half* h_b = new half[N * K];
	float* h_c = new float[M * N];
	float* h_c_ref = new float[M * N];

	/*int blockSize;
	int */
	//Initialize matrices
	for (int i = 0;i < M * K;i++)
		h_a[i] = __float2half(1.0f);
	for (int i = 0;i < N * K;i++)
		h_b[i] = __float2half(1.0f);

	// Device allocations
	half* d_a, *d_b;
	float* d_c;

	cudaMalloc(&d_a, M * K * sizeof(half));
	cudaMalloc(&d_b, K * N * sizeof(half));
	cudaMalloc(&d_c, M * N * sizeof(float));

	//Copy Data to device
	cudaMemcpy(d_a, h_a, M * K * sizeof(half), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, N * K * sizeof(half), cudaMemcpyHostToDevice);
	dim3 gridDim((M + WMMA_M -1)/ WMMA_M, (N + WMMA_N-1) / WMMA_N);
	dim3 blockDim(16,16);  //Warp sized blocks

	wmma_matmul << <gridDim,blockDim >> > (d_a, d_b, d_c);

	cudaMemcpy(h_c, d_c, M * N * sizeof(float), cudaMemcpyDeviceToHost);

	//validate result
	//bool correct = true;
	for (int i = 0;i < M * N;i++)
	{
		std::cout << h_c[i] << " " ;
		if (i % M == 0) std::cout << std::endl;
	}
	//std::cout << (correct ? "Success!" : "Failure") << std::endl;

	delete[] h_a;
	delete[] h_a;
	delete[] h_a;
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	return 0;
}
