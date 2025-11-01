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

struct tiles
{
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
    if (id < total_t)
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
/*
__global__ void __forceinline__ static tensorCoreCsrMatrix(int tpr, int num_v, int total_t, tiles *matrix, float *acc)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < total_t)
    {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
        int col = floor((sqrt(8.0 * id + 1) - 1) / 2);
        int row = id - col * (col + 1) / 2;
        wmma::fill_fragment(c_frag, 0.0f);
        tiles res_t = matrix[id];
        int pos = 0;
#pragma unroll
        for (int i = 0; i < 16; i++)
        {
            u_int16_t v = res_t.tile[pos];
#pragma unroll
            for (int j = 0; j < 16; j++)
            {
                int val = ((v & 1));
                a_frag.x[i * 16 + j] = __int2half_rd(val);
                b_frag.x[j * 16 + i] = __int2half_rd(val);
            }
            pos++;
        }
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        int tile_idx = id; // indice del tile / destinazione
        int ne = c_frag.num_elements;
        for (int e = 0; e < ne; ++e)
        {
            // atomicAdd su float in memoria globale
            atomicAdd(&acc[tile_idx * ne + e], c_frag.x[e]);
        }
    }
}

__global__ __forceinline__ void hadamardProduct(int tpr, int num_v, int total_t, tiles *matrix, float *acc)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < total_t)
    {
        tiles res_t = matrix[id];
        float sum = 0;
        int pos = 0;
#pragma unroll
        for (int i = 0; i < 16; i++)
        {
            u_int16_t v = res_t.tile[pos];
            int index = id * 256 + 16 * i + 1;
            sum += ((v << (16 - i)) & 1) * acc[index];
            pos++;
        }
        printf("%f triangles \n", sum);
    }
}
*/
__global__ void countTriangle(int tpr, int num_v, int total_t, tiles *matrix, float *acc)
{
    int tile_id = blockIdx.x;
    int row = threadIdx.y;
    int col = threadIdx.x;
    int tid = row * 16 + col;
    __shared__ half A[16 * 16];
    __shared__ half B[16 * 16];
    __shared__ float C[16 * 16];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    for (int i = 0; i < tpr; i++)
    {

        tiles res_a = matrix[tile_id];
        tiles res_b = matrix[tile_id];
        u_int16_t rowbits = res_a.tile[row];

        A[tid] = __int2half_ru((rowbits >> (15 - col)) & 1);
        __syncthreads();
        if (tid < 32)
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
            wmma::load_matrix_sync(a_frag, A, 16);
            wmma::load_matrix_sync(b_frag, A, 16);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        __syncthreads();
    }
    if (tid < 32)
        wmma::store_matrix_sync(C, c_frag, 16, wmma::mem_row_major);
    __syncthreads();
    float cval = C[tid];
    float aval = __half2float(A[tid]);
    C[tid] = cval * aval;

    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        float sum = 0.0f;
        for (int i = 0; i < 16; i++)
        {
            for (int j = 0; j < 16; j++)
            {
                float val = C[i * 16 + j];
                sum += val;
            }
        }
        acc[blockIdx.x] = sum;
    }
}

int TTC(int num_v, int n_edges, std::vector<int> offsets, std::vector<int> csr)
{

    cudaSetDevice(0);
    int n_tri = 0;
    int tiles_per_row = ((num_v + 15) >> 4);
    int64_t total_tiles = tiles_per_row * (tiles_per_row + 1) >> 1;
    n_edges = n_edges << 1;
    int padded_size_csr = ((n_edges + 15) >> 4) << 4;
    int *d_csr, *d_ofs, *d_res;
    tiles *d_tiles;
    d_csr = nullptr;
    d_ofs = nullptr;
    d_res = nullptr;
    CHECK(cudaMalloc(&d_csr, (padded_size_csr) * sizeof(int)));
    CHECK(cudaMalloc(&d_ofs, (num_v + 1) * sizeof(int)));
    // tiles_shifted is total_tiles<< 4, because eachtiles contains at most 16 uint_16, so 16² values are stored per thread
    int tiles_shifted = total_tiles;
    CHECK(cudaMalloc(&d_tiles, (tiles_shifted) * sizeof(tiles)));
    // CHECK(cudaMalloc(&d_square, (tiles_shifted) * sizeof(tiles)));
    CHECK(cudaMemcpyAsync(d_csr, csr.data(), n_edges * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyAsync(d_ofs, offsets.data(), (num_v + 1) * sizeof(int), cudaMemcpyHostToDevice));

    dim3 tb_dim_grid((total_tiles + 127) / 128);
    tiles_builder<<<tb_dim_grid, 128>>>(tiles_per_row, num_v, total_tiles, d_csr, d_ofs, d_tiles);
    cudaDeviceSynchronize();
    cudaFree(d_csr);
    cudaFree(d_ofs);

    dim3 blocks_dimension(16, 16);
    dim3 grid_dimension(total_tiles);
    float *d_acc;
    CHECK(cudaMalloc(&d_acc, total_tiles * sizeof(float)));

    countTriangle<<<total_tiles, blocks_dimension>>>(tiles_per_row, num_v, total_tiles, d_tiles, d_acc);
    std::vector<float> res(total_tiles);
    cudaDeviceSynchronize();
    cudaMemcpy(res.data(), d_acc, total_tiles * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_tiles);
    cudaFree(d_acc);
    for (auto i : res)
        n_tri += (int)i;
    return n_tri / 6;
}
