#include "EdgeIterator.h"

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

__device__ __forceinline__ int searchSourceNode(const int *arr, int n, int value)
{
    int low = 0, high = n;
    while (low < high)
    {
        int mid = (low + high) >> 1;
        if (arr[mid] <= value)
            low = mid + 1;
        else
            high = mid;
    }
    return low;
}

__global__ void edge_search_tri(int num_v, int num_e, int *ofs, int *csr, int *results)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < num_e)
    {
        int n_tri = 0;
        int s_node = searchSourceNode(ofs, num_v, id) - 1;
        int d_node = csr[id];
        if (d_node <= s_node)
        {
            results[id] = 0;
            return;
        }
        int s1 = ofs[s_node], e1 = ofs[s_node + 1];
        int s2 = ofs[d_node], e2 = ofs[d_node + 1];
        while (s1 < e1 && s2 < e2)
        {
            int c1 = csr[s1], c2 = csr[s2];
            if (c1 == c2)
            {
                if (c1 > d_node) // Enforce strict ordering s_node < d_node < c1
                {
                    n_tri++;
                }
                s1++;
                s2++;
            }
            else if (c1 < c2)
                s1++;
            else
                s2++;
        }
        results[id] = n_tri;
        return;
    }
}


int SearchTriangle_Edge(int num_v, int n_edges, std::vector<int> &offsets, std::vector<int> &csr, int n_th)
{
    cudaSetDevice(0);
    n_edges = n_edges << 1;
    int *d_csr, *d_ofs, *d_res;
    d_csr = nullptr;
    d_ofs = nullptr;
    d_res = nullptr;

    int n_blocks = (n_edges + (n_th)-1) / (n_th);
    CHECK(cudaMalloc(&d_ofs, (offsets.size()) * sizeof(int)));
    CHECK(cudaMalloc(&d_csr, n_edges * sizeof(int)));
    CHECK(cudaMalloc(&d_res, n_edges * sizeof(int))); // Fix: removed +1
    CHECK(cudaMemcpy(d_ofs, offsets.data(), (num_v + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_csr, csr.data(), (n_edges) * sizeof(int), cudaMemcpyHostToDevice));

    dim3 blockDim(n_th);
    dim3 gridDim(n_blocks);
    edge_search_tri<<<gridDim, blockDim>>>(num_v, n_edges, d_ofs, d_csr, d_res);
    cudaDeviceSynchronize();
    std::vector<int> results(n_edges);                                                       // Fix: changed from num_v to n_edges
    CHECK(cudaMemcpy(results.data(), d_res, n_edges * sizeof(int), cudaMemcpyDeviceToHost)); // Fix: changed size
    cudaFree(d_res);
    cudaFree(d_csr);
    cudaFree(d_ofs);
    int n_tri = 0;
    for (auto i : results)
        n_tri += i;
    return n_tri;
}
