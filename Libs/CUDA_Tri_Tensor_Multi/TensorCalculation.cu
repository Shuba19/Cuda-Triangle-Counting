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

/*
__global__ void wmma_example(half *a, half *b, float *c,
                             int M, int N, int K,
                             float alpha, float beta)
{

    // Leading dimensions. Packed with no transpositions.
    int lda = M;
    int ldb = K;
    int ldc = M;

    // Tile using a 2D grid
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(acc_frag, 0.0f);
}

/*
/Biggest problem is data forma, cpu will need to prepare data to be elaborated for tensors
/
/
*/
__global__ void static tensorCoreCsrMatrix(int n_blocks, int num_v, int *csr, int *offsets, u_int16_t *matrix)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < n_blocks && id == 0)
    {
    }
}

__device__ __forceinline__ int static findPivot(int starting, int end, int len, int *row)
{
}

__global__ __forceinline__ void static tiles_builder(int tpr, int num_v, int total_t, int *csr, int *ofs, u_int16_t *matrix)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < total_t  && id == 0)
    {
        int row = floor((sqrt(8.0 * id + 1) - 1) / 2);
        int col = id - row * (row + 1) / 2;
        int s_x, s_y;
        s_x = col * 16;
        s_y = row * 16;
        int pos = id*4;
        if(s_x == tpr*16 -16 || s_y == tpr*16-16){}
        for (int i = s_y; i < s_y + 16; i++)
        {
            u_int16_t c = 0x0;
            int of1, of2;
            of1 = ofs[i];
            of2 = ofs[i + 1];
            for (int j = s_x; j < s_x + 16; j++)
            {
                int t_s = j==i ? 0:1;
                
                c = c<<1;
                c |= t_s;
            }
            printf("for row %d and th_id %d new c = %d\n",i,id,c);
            matrix[id + pos] = c;
            pos++;
            c = 0x0;
        }
    }
}

int TTC(int num_v, int n_edges, std::vector<int> offsets, std::vector<int> csr)
{
    cudaSetDevice(0);
    int n_tri = 0;
    int n_blocks = 256;
    int tiles_per_row = ((num_v + 15) / 16);
    int64_t total_tiles = tiles_per_row * (tiles_per_row + 1) / 2;
    dim3 blockDim(n_blocks);
    dim3 gridDim((tiles_per_row + n_blocks - 1) / n_blocks);
    /*/flow chart :
        in order  to save space, we have to convert CSR to tiles struct, probably a struct like
        {
            size_t[16*16];  fixed 16*16
            int c,r; /column and row
        }
        can't allocate all in once, due to poor resource,
        need to allocate first tpr column as A(i) and tpr row as B(j) , i,j E [0,tpr-1]
        calculate them Σ A_i * B_j, and repeat for all the matrix in order to calculate C_n tiles, with n E [0, n_tiles -1]
        after, need to decide to apply C hadamard_op A, sum and divide by 6, in order to obtain all the triangles or classical A³
     */
    int *d_csr, *d_ofs, *d_res;
    u_int16_t *d_tiles;
    d_csr = nullptr;
    d_ofs = nullptr;
    d_res = nullptr;
    
    CHECK(cudaMalloc(&d_csr, (num_v) * sizeof(int)));
    CHECK(cudaMalloc(&d_ofs, (num_v + 1) * sizeof(int)));
    //tiles_shifted is total_tiles<< 4, because eachtiles contains at most 16 uint_16, so 16² values are stored per thread
    int tiles_shifted = total_tiles << 4;
    CHECK(cudaMalloc(&d_tiles, (tiles_shifted) * sizeof(u_int16_t)));
    CHECK(cudaMemcpyAsync(d_csr, csr.data(), num_v * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyAsync(d_ofs, offsets.data(), (num_v + 1) * sizeof(int), cudaMemcpyHostToDevice));

    dim3 tb_dim_grid((total_tiles + 127) / 128);
    tiles_builder<<<tb_dim_grid, 128>>>(tiles_per_row, num_v, total_tiles, d_csr, d_ofs, d_tiles);
    std::vector<u_int16_t> res_tiles(total_tiles);
    cudaDeviceSynchronize();
    CHECK(cudaMemcpy(res_tiles.data(), d_tiles, total_tiles * sizeof(u_int16_t), cudaMemcpyDeviceToHost));
    cudaFree(d_csr);
    cudaFree(d_tiles);
    cudaFree(d_ofs);
    return total_tiles;
}
