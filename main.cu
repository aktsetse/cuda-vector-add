#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

__global__ void vectorAdd(const float *a, const float *b, float *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        c[i] = a[i] + b[i];
}

void vectorAddCPU(const float *a, const float *b, float *c, int n)
{
    for (int i = 0; i < n; i++)
        c[i] = a[i] + b[i];
}

int main()
{
    int N = 1 << 20; // ~1 million elements
    size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_a = new float[N];
    float *h_b = new float[N];
    float *h_c = new float[N];
    float *h_c_gpu = new float[N];

    // Initialize data
    for (int i = 0; i < N; i++)
    {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(N - i);
    }

    // --- CPU computation ---
    auto start_cpu = std::chrono::high_resolution_clock::now();
    vectorAddCPU(h_a, h_b, h_c, N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();

    // --- GPU computation ---
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    vectorAdd<<<blocks, threads>>>(d_a, d_b, d_c, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, stop);

    cudaMemcpy(h_c_gpu, d_c, bytes, cudaMemcpyDeviceToHost);

    // --- Verify results ---
    bool correct = true;
    for (int i = 0; i < N; i++)
    {
        if (fabs(h_c[i] - h_c_gpu[i]) > 1e-5)
        {
            correct = false;
            break;
        }
    }

    // --- Output ---
    std::cout << "Results match: " << (correct ? "YES ✅" : "NO ❌") << std::endl;
    std::cout << "CPU time: " << cpu_time << " ms\n";
    std::cout << "GPU time: " << gpu_time << " ms\n";
    std::cout << "Speedup: " << cpu_time / gpu_time << "x\n";

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    delete[] h_c_gpu;
}
