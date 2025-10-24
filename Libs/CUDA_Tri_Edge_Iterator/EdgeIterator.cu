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


__global__ void edge_search_tri(int num_e,int num_tc, int *ofs, int *csr, int *results)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id == 1)
    {
        int n_tri = 0;
        //start a
        int s = id*5, e = s+5;
        for(int i= s; i < e && i < num_e; i++)
        {
            //foreach edge in the node 
        }
        printf("\n");
        results[id] = n_tri;
        return;
    }
}

int SearchTriangle_Edge(int num_v,int n_edges, std::vector<int>& offsets, std::vector<int>& csr, int n_th, int n_elements)
{
    cudaSetDevice(0);
    n_edges = n_edges <<1;
    int *d_csr, *d_ofs, *d_res;
    d_csr = nullptr;
    d_ofs = nullptr;
    d_res = nullptr;

    int n_blocks = (n_edges + (n_th*n_elements) - 1) / (n_th*n_elements);
    int space = n_blocks * n_th;
    CHECK(cudaMalloc(&d_ofs, (offsets.size())*sizeof(int)));
    CHECK(cudaMalloc(&d_csr, n_edges * sizeof(int)));
    CHECK(cudaMalloc(&d_res, n_edges * sizeof(int)/n_elements +1));    
    CHECK(cudaMemcpy(d_ofs, offsets.data(), (num_v + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_csr, csr.data(), (n_edges) * sizeof(int), cudaMemcpyHostToDevice));



    dim3 blockDim(n_th);
    dim3 gridDim(n_blocks);
    edge_search_tri<<<gridDim, blockDim>>>(num_v,n_elements, d_ofs, d_csr, d_res);
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
    
   return 0;
}
