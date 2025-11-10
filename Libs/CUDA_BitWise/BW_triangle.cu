#include "BW_triangle.h"

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
namespace BWISETC
{
    struct tiles
    {
        u_int16_t tile[16];
    };

    __device__ int triangular_col_from_id(int id)
    {
        int col = 0;
        while ((col * (col + 1)) / 2 <= id)
            ++col;
        return col - 1;
    }

    __global__ void tiles_builder(int tpr, int num_v, int total_t, int *csr, int *ofs, tiles *matrix)
    {
        int id = blockDim.x * blockIdx.x + threadIdx.x;
        if (id < total_t)
        {
            int col = floor((sqrt(8.0 * id + 1) - 1) / 2);
            int row = id - col * (col + 1) / 2;
            int s_x = col * 16;
            int s_y = row * 16;
            int pos = 0;
            tiles t_res;
#pragma unroll
            for (int i = 0; i < 16; ++i)
            {
                int y = s_y + i;
                u_int16_t c = 0x0;
                if (y >= num_v)
                {
                    t_res.tile[pos++] = c;
                    continue;
                }
                int of1 = ofs[y];
                int of2 = ofs[y + 1];
#pragma unroll
                for (int j = 0; j < 16; ++j)
                {
                    int x = s_x + j;
                    int t_s = 0;
                    if (x < num_v)
                    {
                        int low = of1;
                        int high = of2 - 1;
                        while (low <= high)
                        {
                            int mid = low + ((high - low) >> 1);
                            if (csr[mid] == x)
                            {
                                t_s = 1;
                                break;
                            }
                            else if (csr[mid] < x)
                            {
                                low = mid + 1;
                            }
                            else
                            {
                                high = mid - 1;
                            }
                        }
                    }
                    c = (c << 1) | (u_int16_t)t_s;
                }
                t_res.tile[pos++] = c;
            }
            matrix[id] = t_res;
        }
    }

    int64_t BWTC(int num_v, int n_edges, std::vector<int> offsets, std::vector<int> csr)
    {

        cudaSetDevice(0);
        int tiles_per_row = ((num_v + 15) >> 4);
        int64_t total_tiles = tiles_per_row * (tiles_per_row + 1) >> 1;
        n_edges = n_edges << 1;
        int padded_size_csr = ((n_edges + 15) >> 4) << 4;
        int *d_csr, *d_ofs;
        tiles *d_tiles;
        d_csr = nullptr;
        d_ofs = nullptr;
        CHECK(cudaMalloc(&d_csr, (padded_size_csr) * sizeof(int)));
        CHECK(cudaMalloc(&d_ofs, (num_v + 1) * sizeof(int)));
        CHECK(cudaMalloc(&d_tiles, (total_tiles) * sizeof(tiles)));
        CHECK(cudaMemcpy(d_csr, csr.data(), n_edges * sizeof(int), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_ofs, offsets.data(), (num_v + 1) * sizeof(int), cudaMemcpyHostToDevice));

        dim3 tb_dim_grid((total_tiles + 127) / 128);
        tiles_builder<<<tb_dim_grid, 1024>>>(tiles_per_row, num_v, total_tiles, d_csr, d_ofs, d_tiles);
        cudaDeviceSynchronize();
        cudaFree(d_csr);
        cudaFree(d_ofs);
        cudaFree(d_tiles);

        int64_t num = 0;

        return num / 6;
    }
}