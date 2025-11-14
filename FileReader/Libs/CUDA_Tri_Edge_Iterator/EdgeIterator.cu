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
                if (c1 > d_node)
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
__global__ void edge_search_tri_directed(int num_v, int num_e, int *ofs, int *csr, int *results)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_e)
        return;

    int n_tri = 0;

    int s_node = searchSourceNode(ofs, num_v, id) - 1;
    int d_node = csr[id];

    int s1 = ofs[s_node];
    int e1 = ofs[s_node + 1];
    int s2 = ofs[d_node];
    int e2 = ofs[d_node + 1];

    while (s1 < e1 && s2 < e2)
    {
        int c1 = csr[s1];
        int c2 = csr[s2];

        if (c1 == c2)
        {
            if (c1 > d_node)
                n_tri++;
            s1++;
            s2++;
        }
        else if (c1 < c2)
        {
            s1++;
        }
        else
        {
            s2++;
        }
    }
    results[id] = n_tri;
}

__global__ void sum_results(int num_e, int *d_res, int *d_sum)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    extern __shared__ int s_data[];
    if (id < num_e)
        s_data[tid] = d_res[id];
    else
        s_data[tid] = 0;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
    {
        if (tid < stride)
            s_data[tid] += s_data[tid + stride];
        __syncthreads();
    }
    if (tid == 0)
        atomicAdd(d_sum, s_data[0]);
}
out_type SearchTriangle_Edge_Iterator(int num_v, int n_edges, std::vector<int> &offsets, std::vector<int> &csr, bool undirected)
{
    cudaSetDevice(0);
    int *d_csr = nullptr, *d_ofs = nullptr, *d_res = nullptr, *d_sum = nullptr;
    n_edges = n_edges<<1;
    int n_blocks = (n_edges + 127) / 128;
    CHECK(cudaMalloc(&d_ofs, (offsets.size()) * sizeof(int)));
    CHECK(cudaMalloc(&d_csr, n_edges * sizeof(int)));
    CHECK(cudaMalloc(&d_res, n_edges * sizeof(int)));
    CHECK(cudaMalloc(&d_sum, sizeof(int)));
    CHECK(cudaMemcpy(d_ofs, offsets.data(), (num_v + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_csr, csr.data(), n_edges * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_sum, 0, sizeof(int)));

    dim3 blockDim(128);
    dim3 gridDim(n_blocks);
    if (undirected)
        edge_search_tri<<<gridDim, blockDim>>>(num_v, n_edges, d_ofs, d_csr, d_res);
    else
        edge_search_tri_directed<<<gridDim, blockDim>>>(num_v, n_edges, d_ofs, d_csr, d_res);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    sum_results<<<gridDim, blockDim, blockDim.x * sizeof(int)>>>(n_edges, d_res, d_sum);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    int tri_count = 0;
    CHECK(cudaMemcpy(&tri_count, d_sum, sizeof(int), cudaMemcpyDeviceToHost));

    cudaFree(d_res);
    cudaFree(d_csr);
    cudaFree(d_ofs);
    cudaFree(d_sum);

    int64_t n_tri = tri_count;
     if (!undirected) n_tri /= 3;

    return n_tri;
}
