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

struct tiles{
    u_int16_t tile[16];
};

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
__global__ __forceinline__ void static tiles_builder(int tpr, int num_v, int total_t, int *csr, int *ofs, tiles *matrix)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < total_t )
    {
        int col = floor((sqrt(8.0 * id + 1) - 1) / 2);
        int row = id - col * (col + 1) / 2;
        int s_x, s_y;
        s_x = col * 16;
        s_y = row * 16;
        int pos = 0;
        // Each thread handles one tile of 16 rows (16 * u_int16_t)
        // if (col == tpr - 1)
        //     if (row == tpr - 1)
        //         printf("last diagonal reached\n");
        //     else
        //         printf("y wall encountered row %d id_th %d \n", row, id);
        // if (row == tpr - 1)
        //     printf("x wall encountered  col %d id_th %d \n", col, id);
        tiles t_res;
        for (int i = s_y; i < s_y + 16; i++)
        {
            u_int16_t c = 0x0;
            int of1, of2;
            of1 = ofs[i];
            of2 = ofs[i + 1];
            for (int j = s_x; j < s_x + 16; j++)
            {
                int t_s = 0;
                if (i < num_v && j < num_v)
                {
                    int low = of1;
                    int high = of2 - 1;
                    while (low <= high)
                    {
                        int mid = low + (high - low) / 2;
                        if (csr[mid] == j)
                        {
                            t_s = 1;
                            break;
                        }
                        else if (csr[mid] < j)
                        {
                            low = mid + 1;
                        }
                        else
                        {
                            high = mid - 1;
                        }
                    }
                }
                c = c << 1;
                c |= t_s;
            }
            t_res.tile[pos++] = c;
        }
        matrix[id] = t_res;
    }
}

__global__ void static tensorCoreCsrMatrix(int n_blocks, int num_v, int *csr, int *offsets, u_int16_t *matrix)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < n_blocks)
    {
        __shared__    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> b_frag;
        __shared__    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
        __shared__ wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
        
    }
}

int TTC(int num_v, int n_edges, std::vector<int> offsets, std::vector<int> csr)
{
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
    cudaSetDevice(0);
    int n_tri = 0;
    int n_blocks = 256;
    int tiles_per_row = ((num_v + 15) >> 4);
    int64_t total_tiles = tiles_per_row * (tiles_per_row + 1) >> 1;
    dim3 blockDim(n_blocks);
    dim3 gridDim((tiles_per_row + n_blocks - 1) / n_blocks);
    n_edges = n_edges << 1;
    int padded_size_csr = ((n_edges + 15) >> 4) << 4;
    int *d_csr, *d_ofs, *d_res;
    u_int16_t *d_square;
    tiles *d_tiles;
    d_csr = nullptr;
    d_ofs = nullptr;
    d_res = nullptr;
    d_square = nullptr;
    CHECK(cudaMalloc(&d_csr, (padded_size_csr) * sizeof(int)));
    CHECK(cudaMalloc(&d_ofs, (num_v + 1) * sizeof(int)));
    // tiles_shifted is total_tiles<< 4, because eachtiles contains at most 16 uint_16, so 16² values are stored per thread
    int tiles_shifted = total_tiles;
    CHECK(cudaMalloc(&d_tiles, (tiles_shifted) * sizeof(tiles)));
    //CHECK(cudaMalloc(&d_square, (tiles_shifted) * sizeof(tiles)));
    CHECK(cudaMemcpyAsync(d_csr, csr.data(), n_edges * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyAsync(d_ofs, offsets.data(), (num_v + 1) * sizeof(int), cudaMemcpyHostToDevice));

    dim3 tb_dim_grid((total_tiles + 127) / 128);
    tiles_builder<<<tb_dim_grid, 128>>>(tiles_per_row, num_v, total_tiles, d_csr, d_ofs, d_tiles);
    cudaDeviceSynchronize();
    cudaFree(d_csr);
    cudaFree(d_ofs);
    cudaFree(d_tiles);
    cudaFree(d_square);
    return total_tiles;
}
