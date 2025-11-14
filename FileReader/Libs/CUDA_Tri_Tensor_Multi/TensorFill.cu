#include "TensorCalculation.h"
#include <algorithm> // <<< FIX: Needed for std::max and std::min (or use CUDA's max/min)
#include <cmath>     // <<< FIX: Included for sqrt/floor, though likely in your .h

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

__global__ void d_fill_sup(int n_edges, int num_v, int *csr, int *ofs, tiles *sup)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < n_edges)
    {
        int col = csr[id];
        int row = searchSourceNode(ofs, num_v + 1, id) - 1;
        if (row < 0 || col < 0 || row >= num_v || col >= num_v) return;

        // Map to triangular tile index
        int tile_row = row >> 4; // row / 16
        int tile_col = col >> 4; // col / 16
        if (tile_row > tile_col) return;

        int block_id = tile_row + (tile_col * (tile_col + 1)) / 2;

        int b_r = row & 15;      
        int b_c = col & 15;      
        sup[block_id].tile[b_r * 16 + b_c] = 1.0;
    }
}

__global__ void d_calc_square(int tpr, int num_v, tiles *sup, tiles *square)
{
    int tile_id = blockIdx.x;
    int row = threadIdx.y;
    int col = threadIdx.x;
    int tid = row * 16 + col;

    __shared__ half A[256];
    __shared__ half B[256];
    __shared__ double C[256];
    __shared__ float temp_C[256];

    int t_col = triangular_col_from_id(tile_id);
    int t_row = tile_id - t_col * (t_col + 1) / 2;
    C[tid] = 0.0;
    temp_C[tid] = 0.0f;


#pragma unroll
    for (int k = 0; k < tpr; k++)
    {
        int r1 = max(t_col, k);
        int c1 = min(t_col, k);
        int id1 = r1 * (r1 + 1) / 2 + c1;

        int r2 = max(k, t_row);
        int c2 = min(k, t_row);
        int id2 = r2 * (r2 + 1) / 2 + c2;
        tiles a_tile, b_tile;
        A[tid] = __double2half(sup[id1].tile[tid]);
        B[tid] = __double2half(sup[id2].tile[tid]);
        __syncthreads();
        if (tid < 32)
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

            wmma::load_matrix_sync(a_frag, A, 16);
            wmma::load_matrix_sync(b_frag, B, 16);
            wmma::fill_fragment(c_frag, 0.0f);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            wmma::store_matrix_sync(temp_C, c_frag, 16, wmma::mem_row_major);
        }
        __syncthreads();
        C[tid] += (double)temp_C[tid];
        __syncthreads();
    }
    square[tile_id].tile[tid] = C[tid];
}


__global__ void d_calc_cube(int tpr, int num_v, tiles *sup, tiles *square, int *diag)
{
    int tile_id = blockIdx.x;
    int row = threadIdx.y;
    int col = threadIdx.x;
    int tid = row * 16 + col;

    __shared__ half A[256];
    __shared__ half B[256];
    __shared__ double C[256];
    __shared__ float temp_C[256];

    int t_col = triangular_col_from_id(tile_id);
    int t_row = tile_id - t_col * (t_col + 1) / 2;
    C[tid] = 0.0;
    temp_C[tid] = 0.0f;


#pragma unroll
    for (int k = 0; k < tpr; k++)
    {
        int r1 = max(t_col, k);
        int c1 = min(t_col, k);
        int id1 = r1 * (r1 + 1) / 2 + c1;

        int r2 = max(k, t_row);
        int c2 = min(k, t_row);
        int id2 = r2 * (r2 + 1) / 2 + c2;
        tiles a_tile, b_tile;
        A[tid] = __double2half(sup[id1].tile[tid]);
        B[tid] = __double2half(sup[id2].tile[tid]);
        __syncthreads();
        if (tid < 32)
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

            wmma::load_matrix_sync(a_frag, A, 16);
            wmma::load_matrix_sync(b_frag, B, 16);
            wmma::fill_fragment(c_frag, 0.0f);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            wmma::store_matrix_sync(temp_C, c_frag, 16, wmma::mem_row_major);
        }
        __syncthreads();
        C[tid] += (double)temp_C[tid];
        __syncthreads();
    }
    square[tile_id].tile[tid] = C[tid];
}


out_type TTC_v2(int num_v, int n_edges,const std::vector<int>& offsets, const std::vector<int>& csr)
{
    int sub_block = (num_v + 15) >> 4;
    n_edges = n_edges << 1; 

    int *d_csr, *d_ofs;
    tiles *d_sup, *d_square;
    int64_t total_tiles = sub_block * (sub_block + 1) >> 1;

    CHECK(cudaMalloc(&d_csr, n_edges * sizeof(int)));
    CHECK(cudaMalloc(&d_ofs, (num_v + 1) * sizeof(int)));
    CHECK(cudaMalloc(&d_sup, total_tiles * sizeof(tiles)));
    CHECK(cudaMemset(d_sup, 0, total_tiles * sizeof(tiles)));

    CHECK(cudaMemcpy(d_csr, csr.data(), n_edges * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_ofs, offsets.data(), (num_v + 1) * sizeof(int), cudaMemcpyHostToDevice));

    int n_th = 128;
    dim3 blockDimFill(n_th);
    dim3 gridDimFill((n_edges + n_th - 1) / n_th);
    d_fill_sup<<<gridDimFill, blockDimFill>>>(n_edges, num_v, d_csr, d_ofs, d_sup);

    CHECK(cudaMalloc(&d_square, total_tiles * sizeof(tiles)));
    CHECK(cudaMemset(d_square, 0, total_tiles * sizeof(tiles)));
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    dim3 blockDim(16,16);
    dim3 gridDim(total_tiles);
    cudaFree(d_csr);
    cudaFree(d_ofs);
    d_calc_square<<<gridDim,blockDim>>>(sub_block,num_v,d_sup,d_square);
    cudaDeviceSynchronize();
    cudaFree(d_csr);
    cudaFree(d_ofs);
    cudaFree(d_sup);
    cudaFree(d_square);
    return 0;
}
