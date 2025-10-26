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

const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

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
__global__ void tensorCoreCsrMatrix(int n_blocks, int *csr, int *offsets)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < n_blocks)
    {
    }
}

int TTC(int num_v, int n_edges, std::vector<int> offsets, std::vector<int> csr)
{
    cudaSetDevice(0);
    int n_tri = 0;
    int n_blocks = 1 << 8;
    int tiles_per_row = ((num_v + 15) / 16);
    int n_tiles = tiles_per_row * tiles_per_row;
    dim3 blockDim(n_blocks);
    dim3 gridDim((num_v + n_blocks - 1) / n_blocks);
    /*/flow chart :
        in order  to save space, we have to convert CSR to tiles struct, probably a struct like
        {
            size_t[16*16];  fixed 16*16
            int c,r; /column and row
        }
        can't allocate all in once, due to poor resource,
        need to allocate first tpr column as A(i) and tpr row as B(j) , i,j E [0,tpr-1]
        calculate them Σ A_i * B_j, and repeat for all the matrix in order to calculate C_n tiles, with n E [0, n_tiles -1]
        after, apply C hadamard_op A, sum and divide by 6, in order to obtain all the triangles

        A
     */
    for(int i = 0; i< n_tiles; i++ )
    {
        //prepare first row
        std::vector<int1> tiles(n_blocks);
        //do something

    }
    int *d_csr, *d_ofs, *d_res;
    d_csr = nullptr;
    d_ofs = nullptr;
    d_res = nullptr;

    cudaDeviceSynchronize();
    return n_tri;
}
