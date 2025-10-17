#include "TriangleCounting.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__device__ bool search_Tri(int *row, int len, int goal)
{
    if (len <= 0)
        return false;
    int lo = 0;
    int hi = len - 1;
    while (lo <= hi)
    {
        int mid = lo + ((hi - lo) >> 1);
        int val = row[mid];
        if (val == goal)
            return true;
        if (val < goal)
            lo = mid + 1;
        else
            hi = mid - 1;
    }
    return false;
}

__global__ void printfGPU(int num_v, int *csr_size, int **csr, int *results)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (id < num_v)
    {
        results[id] = 0;
        int num_childs = csr_size[id];
        
        for (int i = 0; i < num_childs; i++)
        {
            int v = csr[id][i];
            if (v <= id || v >= num_v) continue;
            
            int v_size = csr_size[v];
            for (int j = 0; j < v_size; j++)
            {
                int w = csr[v][j];
                if (w <= v || w >= num_v) continue;
                bool res = search_Tri(csr[id], num_childs, w);
                if (res)
                {
                    results[id]++;
                }
            }
        }
    }
}

void printCSR(int num_v, int edges, int *csr_size, int **csr)
{
    cudaSetDevice(0);
    int threads_per_block = 256;
    int n_blocks = (num_v + threads_per_block - 1) / threads_per_block;
    dim3 grid(n_blocks);
    dim3 blocks(threads_per_block);

    int *d_csr_size = nullptr;
    cudaMalloc(&d_csr_size, num_v * sizeof(int));
    cudaMemcpy(d_csr_size, csr_size, num_v * sizeof(int), cudaMemcpyHostToDevice);
    
    std::vector<int *> h_row_ptrs(num_v, nullptr);
    for (int i = 0; i < num_v; i++)  
    {
        int len = csr_size[i];
        if (len > 0 && csr[i] != nullptr)
        {
            int *d_row = nullptr;
            cudaMalloc(&d_row, len * sizeof(int));
            cudaMemcpy(d_row, csr[i], len * sizeof(int), cudaMemcpyHostToDevice);
            h_row_ptrs[i] = d_row;
        }
        else
        {
            h_row_ptrs[i] = nullptr;
        }
    }
    
    int **d_csr = nullptr;
    cudaMalloc(&d_csr, num_v * sizeof(int *));  
    cudaMemcpy(d_csr, h_row_ptrs.data(), num_v * sizeof(int *), cudaMemcpyHostToDevice);

    int *d_res = nullptr;
    cudaMalloc(&d_res, num_v * sizeof(int)); 
    
    printfGPU<<<grid, blocks>>>(num_v, d_csr_size, d_csr, d_res);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
    }
    
    cudaDeviceSynchronize();
    std::vector<int> results(num_v);
    cudaMemcpy(results.data(), d_res, num_v * sizeof(int), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < num_v; i++) 
    {
        if (h_row_ptrs[i])
            cudaFree(h_row_ptrs[i]);
    }

    cudaFree(d_csr);
    cudaFree(d_csr_size);
    cudaFree(d_res);
    int sum = 0;
    for (auto i : results)
        sum += i;
    std::cout << "Completed with " << sum << " triangles" << std::endl;
}