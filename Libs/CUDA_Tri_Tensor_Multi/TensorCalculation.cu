#include "TensorCalculation.h"
using namespace nvcuda;
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

__global__ void tensorCoreCsrMatrix(int num_v, int *csr, int *offsets)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < num_v && id == 0)
    {
        half A[16 * 16];
        for (int i = 0; i < 16; i++)
        {

            int ofs = offsets[i];
            int of2 = offsets[i + 1];
            for (int j = ofs; (j < of2) && (j - ofs) < 16; j++)
            {
                if (csr[j] >= (16 * (id + 1)))
                    break;
                A[i * 16 + csr[j]] = 1;
            }
        }
    }
}

int MatrixMulti(int num_v, int n_edges, std::vector<int> offsets, std::vector<int> csr, int n_blocks)
{
    cudaSetDevice(0);
    int n_tri = 0;
    dim3 blockDim(n_blocks);
    dim3 gridDim((num_v + n_blocks - 1) / n_blocks);
    int *d_csr, *d_ofs, *d_res;
    d_csr = nullptr;
    d_ofs = nullptr;
    d_res = nullptr;
    CHECK(cudaMalloc(&d_ofs, (offsets.size()) * sizeof(int)));
    CHECK(cudaMalloc(&d_csr, n_edges * sizeof(int)));
    CHECK(cudaMalloc(&d_res, num_v * sizeof(int)));
    CHECK(cudaMemcpy(d_ofs, offsets.data(), (num_v + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_csr, csr.data(), (n_edges) * sizeof(int), cudaMemcpyHostToDevice));
    tensorCoreCsrMatrix<<<gridDim, blockDim>>>(num_v, d_csr, d_ofs);
    cudaDeviceSynchronize();
    return n_tri;
}


