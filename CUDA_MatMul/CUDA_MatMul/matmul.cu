#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <iostream>

#define BLOCK_SIZE 32
#define TILE_WIDTH 32
#define COARSE_FACTOR 32

void randomize_matrix(float* a, int N)
{	
	srand(unsigned(time(NULL)));
	for (int i = 0;i < N;i++)
	{
		for (int j = 0;j < N;j++)
		{
			float temp = (float)(rand() % 5) + 0.01 * (rand() % 5);
			temp = (rand() % 2 == 0) ? temp : -temp;
			a[i*N+j] = temp;
		}
	}	
}

void print_matrix(float* a, int width)
{
	for (int i = 0;i < width;i++)
	{
		for (int j = 0;j < width;j++)
		{
			std::cout << a[i * width + j] << " ";
		}
		std::cout << std::endl;
	}
}

//template <const int BM, const int BN, const int BK, const int TM, const int TN>
//__global__ void __launch_bounds__((BM *BN)/(TM*TN),1) 
//    matmul_2DBlocktiling_smem_reg(float* a, float* b, float* c, int width)
//{
//	const int cRow = blockIdx.y;
//	const int cCol = blockIdx.x;
//
//	const int totalResultsBlockTile = BM * BN;
//	const int numThreadsBlockTile = totalResultsBlockTile / (TM * TN);
//	assert(numThreadsBlockTile == blockDim.x);
//
//	const int threadCol = threadIdx.x % (BN / TN);
//	const int threadRow = threadIdx.x / (BN / TN);
//
//	__shared__ float a_s[BM * BK];
//	__shared__ float b_s[BK * BN];
//	
//
//}


__global__ void matmul_PMPPThreadCoarsening(float* a, float* b, float* c, int width)
{
	__shared__ float a_s[TILE_WIDTH][TILE_WIDTH];
	__shared__ float b_s[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;

	//identify the row and column of the c element to be worked for
	int Row = by * TILE_WIDTH + ty;
	int colStart = bx * TILE_WIDTH * COARSE_FACTOR + tx;
	
	float sum[COARSE_FACTOR];
	for (int c_i = 0; c_i <COARSE_FACTOR; ++c_i)
		sum[c_i] = 0.0f;

	//Loop over a and b to load into smem; 
	for (int ph = 0;ph < width / TILE_WIDTH;ph++)
	{
		a_s[ty][tx] = a[Row * width + ph * TILE_WIDTH + tx];
		for (int c_i = 0;c_i < COARSE_FACTOR;c_i++)
		{
			int col = colStart + c_i * TILE_WIDTH;
			b_s[ty][tx] = b[(ph * TILE_WIDTH + ty) * width + col];
			__syncthreads();

			for (int k = 0;k < TILE_WIDTH;k++)
				sum[k] += a_s[ty][k] * b_s[k][tx];
			__syncthreads();

		}
	}
	for (int i = 0;i < COARSE_FACTOR;i++)
	{
		int col = colStart + i * TILE_WIDTH;
		c[Row * width + col] = sum[i];
	}
}
__global__ void matmul_sharedmem_blocking(float* a, float* b, float* c, int width)
{
	const int cRow = blockIdx.x;
	const int cCol = blockIdx.y;

	__shared__ float a_s[BLOCK_SIZE * BLOCK_SIZE];
	__shared__ float b_s[BLOCK_SIZE * BLOCK_SIZE];

	const int threadRow = threadIdx.x / BLOCK_SIZE;
	const int threadCol = threadIdx.x % BLOCK_SIZE;

	a += cRow * BLOCK_SIZE * width;
	b += cCol * BLOCK_SIZE;
	c += cRow * BLOCK_SIZE * width + cCol * BLOCK_SIZE;

	float sum = 0.0f;

	for (int bkIdx = 0; bkIdx < width; bkIdx+=BLOCK_SIZE)
	{
		a_s[threadRow * BLOCK_SIZE + threadCol] = a[threadRow * width + threadCol];
		b_s[threadRow * BLOCK_SIZE + threadCol] = b[threadRow * width + threadCol];
		__syncthreads();

		a += BLOCK_SIZE;
		b += BLOCK_SIZE * width;

		for (int dotIdx = 0;dotIdx < BLOCK_SIZE;dotIdx++)
		{
			sum += a_s[threadRow * BLOCK_SIZE + dotIdx] * b_s[dotIdx * BLOCK_SIZE + threadCol];
		}
		__syncthreads();
	}
	c[threadRow * width + threadCol] += sum;

}

__global__ void matmul_tiled_smem(float* a, float* b, float* c, int width)
{
	__shared__ float a_s[TILE_WIDTH][TILE_WIDTH];
	__shared__ float b_s[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;

	//c element row/col calculations
	int Row = by * TILE_WIDTH + ty;
	int Col = bx * TILE_WIDTH + tx;

	float sum = 0.0f;
	for (int ph = 0;ph < width / TILE_WIDTH;ph++)
	{
		//Loading into smem 
		a_s[ty][tx] = a[Row * width + ph * TILE_WIDTH + tx];
		b_s[ty][tx] = b[(ph * TILE_WIDTH + ty) * width + Col];
		
		__syncthreads();

		for (int k = 0;k < TILE_WIDTH;k++)
		{
			sum += a_s[ty][k] * b_s[k][tx];
		}
		__syncthreads();
	}
	c[Row * width + Col] = sum;
}
__global__ void matmul_globalmemcoalescing(float* a, float* b, float* c, int width)
{
	const int row = blockIdx.x * BLOCK_SIZE + (threadIdx.x / BLOCK_SIZE); 
	const int col = blockIdx.y * BLOCK_SIZE + (threadIdx.x % BLOCK_SIZE);

	if ((row < width) && (col < width))
	{
		float sum = 0.0f;
		for (int k = 0;k < width;++k)
		{
			sum += a[row * width + k] * b[k * width + col];
		}
		c[row * width + col]= sum;

	}
}

__global__ void matmul_kernel(float* a, float* b, float* c, int width)
{
	//width = 256, 512, 1024, 2048, 4096 block(16, 16)(32, 32)
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if ((row < width) && (col < width))
	{
		float sum = 0.0f;
		for (int k = 0;k < width;++k)
		{
			sum += a[row * width + k] * b[k * width + col];
		}
		c[row * width + col] = sum;
	}
}

void matrixMultiplyCPU(float* a, float* b, float* c, int N)
{
	for (int row = 0;row < N;row++)
	{
		for (int col = 0; col < N;col++)
		{
			float sum = 0.0f;
			for (int k = 0;k < N;k++)
			{
				sum += a[row * N + k] * b[k * N + col];
			}
			c[row * N + col] = sum;
		}
	}
}

int main()
{
	float* a_h, * b_h, * c_h;// * c_h_cpu;
	float* a_d, *b_d, *c_d;
	const int width =512; 
	//prop.maxThreadsPerBlock : 1024 is the number;
	size_t size = width * width * sizeof(float);

	a_h = new float[width * width];
	b_h = new float[width * width];
	c_h = new float[width * width];
	//c_h_cpu = new float[width * width];


	randomize_matrix(a_h, width);
	//print_matrix(a_h, width);

	randomize_matrix(b_h, width);
	//print_matrix(b_h, width);

	cudaMalloc(&a_d, size);
	cudaMalloc(&b_d, size);
	cudaMalloc(&c_d, size);
	
	cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice);
	

	//for matmul_kernel,tiled_smem,PMPPThreadCoarsening; 
	dim3 block(32,32);
	dim3 grid((width + block.x - 1) / block.x, (width + block.y - 1) / block.y);

	//for matmul_globalmemcoalescing
	/*dim3 grid(ceil(width/32), ceil(width/32));
	dim3 block(16*16);*/
	//for matmul_sharedmem_blocking
	/*dim3 grid(ceil(width / 32), ceil(width / 32));
	dim3 block(32 * 32);*/
	
	//matmul_kernel <<<grid, block >>> (a_d, b_d, c_d, width);
	//matmul_globalmemcoalescing <<<grid, block >>> (a_d, b_d, c_d, width);
	matmul_tiled_smem << <grid, block >> > (a_d, b_d, c_d, width);
	//matmul_sharedmem_blocking << <grid, block >> > (a_d, b_d, c_d, width);
	//matmul_PMPPThreadCoarsening <<<grid, block >>>(a_d, b_d, c_d, width);
	//matmul_2DBlocktiling_smem_reg<<<grid,block>>>(a_d,b_d,c_d,width);


	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Kernel launch error: %s\n", cudaGetErrorString(err));
	}
	cudaDeviceSynchronize();
	


	cudaMemcpy(c_h, c_d, size, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	
	//matrixMultiplyCPU(a_h, b_h, c_h_cpu, width);
	
	print_matrix(c_h, width);

	//std::cout << "Result Matrix from CPU is: " << std::endl;
	//print_matrix(c_h_cpu, width);

	cudaFree(a_d);
	cudaFree(b_d);
	cudaFree(c_d);

	delete a_h;
	delete b_h;
	delete c_h;
	//delete c_h_cpu;
	return 0;

}