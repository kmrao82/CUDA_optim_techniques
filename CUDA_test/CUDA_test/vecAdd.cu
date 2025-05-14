#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

__global__ void vecAdd(float* a, float* b, float* c, int num) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int N = 1024;
    std::vector<float> a(N), b(N), c(N); // Resized to N

    for (int i = 0; i < N; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    float* a_d, * b_d, * c_d;
    cudaMalloc(&a_d, N * sizeof(float));
    cudaMalloc(&b_d, N * sizeof(float));
    cudaMalloc(&c_d, N * sizeof(float));

    cudaMemcpy(a_d, a.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int maxThreadsPerBlock = prop.maxThreadsPerBlock;
    int gridSize = (N + maxThreadsPerBlock - 1) / maxThreadsPerBlock;

    std::cout << "Kernel params: Threads per block " << maxThreadsPerBlock
        << " grid size " << gridSize << std::endl;

    vecAdd <<<gridSize, maxThreadsPerBlock >>> (a_d, b_d, c_d, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();

    cudaMemcpy(c.data(), c_d, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        std::cout << c[i] << " ";
    }

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);

    return 0;
}
