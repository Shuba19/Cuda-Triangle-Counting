    #include "BitWiseCalculation.h"
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


    __global__ static void BuildBWMatrix(int total_tiles, int tpr, int num_v, int *csr, int *offsets, u_int16_t *matrix)
    {
        int id = blockDim.x * blockIdx.x + threadIdx.x;
        if (id < total_tiles && id ==0)
        {
        //each thread has to compute a tile of the matrix shifting 1 or 0 its own matrix tile if there is or not and edge between u,v
        //the id to access the matrix is computed in this way: a thread th_i will access to the matrix[i]'s cell with its own idea, but to build 
        //the tile he must access it's own row and eventually has to shift x cells for x row
        }
    }

    __global__ static void BitWiseCoreCsrMatrix(int n_blocks, int num_v, int *csr, int *offsets, u_int16_t *matrix)
    {
        int id = blockDim.x * blockIdx.x + threadIdx.x;
        if (id < n_blocks && id ==0)
        {
            
        }
    }

    int BWC(int num_v, int n_edges, std::vector<int> offsets, std::vector<int> csr)
    {
        cudaSetDevice(0);
        int n_tri = 0;
        int n_blocks = 128;
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
        u_int16_t* d_tiles;
        d_csr = nullptr;
        d_ofs = nullptr;
        d_res = nullptr;
        CHECK(cudaMalloc(&d_csr, num_v * sizeof(int)));
        CHECK(cudaMalloc(&d_tiles, total_tiles * sizeof(u_int16_t)));
        BuildBWMatrix<<<gridDim,blockDim>>>(total_tiles,tiles_per_row,num_v, d_csr,d_ofs,d_tiles);
        cudaDeviceSynchronize();
        cudaFree(d_csr);
        cudaFree(d_tiles);
        return total_tiles;
    }
