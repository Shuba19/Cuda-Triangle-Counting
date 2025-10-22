#include "TriangleCounting.h"

#define CHECK(call)                                                         \
    {                                                                       \
        const cudaError_t error = call;                                     \
        if (error != cudaSuccess)                                           \
        {                                                                   \
            printf("Error %s : %d\n", __FILE__, __LINE__);                  \
            printf("code:%d, reason:%s", error, cudaGetErrorString(error)); \
            exit(1);                                                        \
        }                                                                   \
    }

__device__ bool bin_search_opt(int goal, int *v, int len)
{
    int l = 0;
    int h = len - 1;
    while (l <= h)
    {
        int mid = (l + h) / 2;
        if (v[mid] < goal)
        {
            l = mid + 1;
        }
        else if (v[mid] > goal)
        {
            h = mid -1;
        }
        else
        {
            return true;
        }
    }
    return false;
}

__global__ void d_search_tri(int num_v, int *ofs, int *csr, int *results)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < num_v)
    {
        for (int i = ofs[id]; i < ofs[id + 1]; i++)
        {
            if (csr[i] < id)
                continue;
            int index = csr[i];
            for (int j = ofs[index]; j < ofs[index + 1]; j++)
            {
                int pivot = csr[j];
                if (pivot < index)
                    continue;
               results[id] += bin_search_opt(id, &csr[ofs[pivot]], ofs[pivot+1] -ofs[pivot]) ? 1 : 0;
            }
        }
    }
}

int SearchTriangle(int num_v,int n_edges, std::vector<int>& offsets, std::vector<int>& csr, int n_blocks)
{
    cudaSetDevice(0);
    dim3 blockDim(n_blocks);
    dim3 gridDim((num_v + n_blocks - 1) / n_blocks);
    int *d_csr, *d_ofs, *d_res;
    d_csr = nullptr;
    d_ofs = nullptr;
    d_res = nullptr;
    CHECK(cudaMalloc(&d_ofs, (offsets.size())*sizeof(int)));
    CHECK(cudaMalloc(&d_csr, n_edges * sizeof(int)));
    CHECK(cudaMalloc(&d_res, num_v * sizeof(int)));
    CHECK(cudaMemcpy(d_ofs, offsets.data(), (num_v + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_csr, csr.data(), (n_edges) * sizeof(int), cudaMemcpyHostToDevice));
    d_search_tri<<<gridDim, blockDim>>>(num_v, d_ofs, d_csr, d_res);
    cudaDeviceSynchronize();
    std::vector<int> results(num_v);
    CHECK(cudaMemcpy(results.data(), d_res, num_v * sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(d_res);
    cudaFree(d_csr);
    cudaFree(d_ofs);
    int n_tri = 0;
    for (auto i : results)
        n_tri += i;
    return n_tri;
}
