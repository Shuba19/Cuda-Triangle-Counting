#include "FileReader/FileReader.h"
#include "Libs/CUDA_Tri_Node_Iterator/NodeIterator.h"
#include "Libs/CUDA_Tri_Edge_Iterator/EdgeIterator.h"
#include "Libs/CUDA_Tri_Tensor_Multi/TensorCalculation.h"
#include "Libs/CUDA_Tri_BitWise_Operation/BitWiseCalculation.h"
#define REPS 1
#define V1 true
#define V2 true
#define V3 true
#define V4 false
int main(int argc, char *argv[])
{
    GraphFR FR(argv[1]);
    std::cout << "Reading Graph :" << argv[1] << "  with a density of:" << float(FR.num_edge / FR.num_v) << std::endl;
    SearchTriangle_Edge(FR.num_v, FR.num_edge, FR.offsets, FR.csr, 128);
    int64_t result;
    cudaEvent_t t1, t2;
    cudaEventCreate(&t1);
    cudaEventCreate(&t2);
    float elaps = 0.0f;
    if (V1)
    {
        cudaEventRecord(t1);
        for (int i = 0; i < REPS; i++)
            result = SearchTriangle_Edge(FR.num_v, FR.num_edge, FR.offsets, FR.csr, 128);
        cudaEventRecord(t2);
        cudaEventSynchronize(t2);
        cudaEventElapsedTime(&elaps, t1, t2);
        elaps = elaps / REPS;
        std::cout << "Elapsed time: " << elaps << " ms with " << result << " triangles with Edge Iterator." << std::endl;
    }
    if (V2)
    {
        cudaEventRecord(t1);
        for (int i = 0; i < REPS; i++)
            result = SearchTriangle(FR.num_v, FR.num_edge, FR.offsets, FR.csr, 128);
        cudaEventRecord(t2);
        cudaEventSynchronize(t2);
        cudaEventElapsedTime(&elaps, t1, t2);
        elaps = elaps / REPS;
        std::cout << "Elapsed time: " << elaps << " ms with " << result << " triangles with Node Iterator." << std::endl;
    }
    if (V3)
    {
        cudaEventRecord(t1);
        for (int i = 0; i < REPS; i++)
            result = TTC(FR.num_v, FR.num_edge, FR.offsets, FR.csr);
        cudaEventRecord(t2);
        cudaEventSynchronize(t2);
        cudaEventElapsedTime(&elaps, t1, t2);
        elaps = elaps / REPS;
        std::cout << "Elapsed time: " << elaps << " ms with " << result << " triangles with TTC method." << std::endl;
    }
    if (V4)
    {
        cudaEventRecord(t1);
        for (int i = 0; i < REPS; i++)
            result = BWC(FR.num_v, FR.num_edge, FR.offsets, FR.csr);
        cudaEventRecord(t2);
        cudaEventSynchronize(t2);
        cudaEventElapsedTime(&elaps, t1, t2);
        elaps = elaps / REPS;
        std::cout << "Elapsed time: " << elaps << " ms with " << result << " triangles with Bitwise method." << std::endl;
    }

    return 0;
}