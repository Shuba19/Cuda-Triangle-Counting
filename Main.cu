#include "FileReader/command_args.h"
#include "FileReader/FileReader.h"
#include "Libs/CUDA_Tri_Node_Iterator/NodeIterator.h"
#include "Libs/CUDA_Tri_Edge_Iterator/EdgeIterator.h"
#include "Libs/CUDA_Tri_Tensor_Multi/TensorCalculation.h"
#include "Libs/CUDA_Hybrid_Operation/CUDA_Hybrid_Operation.h"
#include "Libs/CUDA_BitWise/BW_triangle.h"
#define REPS 1
#define V1 true
#define V2 true
#define V3 false
#define V4 false
int main(int argc, char *argv[])
{
    CommandArgs ca = parse_command_args(argc, argv);
    GraphFR FR(ca.input_file);
    std::cout << "Reading Graph :" << ca.input_file << std::endl;

    cudaEvent_t t1, t2;
    float elaps = 0.0f;
    cudaEventCreate(&t1);
    cudaEventCreate(&t2);
    cudaEventRecord(t1);
    int64_t result = SearchTriangle_Edge(FR.num_v, FR.num_edge, FR.offsets, FR.csr, ca.undirect);
    cudaEventRecord(t2);
    cudaEventSynchronize(t2);
    cudaEventElapsedTime(&elaps, t1, t2);
    if (ca.benchmark == false)
    {
        std::cout << "Graph read with " << FR.num_v << " vertices and " << FR.num_edge << " edges in " << elaps << " ms." << std::endl;
        std::cout << "Initial triangle count with Edge Iterator: " << result << std::endl;
        return 0;
    }
    if (V1)
    {
        cudaEventRecord(t1);
        for (int i = 0; i < REPS; i++)
            result = SearchTriangle_Edge(FR.num_v, FR.num_edge, FR.offsets, FR.csr, ca.undirect);
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
            result = SearchTriangle(FR.num_v, FR.num_edge, FR.offsets, FR.csr, ca.undirect);
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
            result = BWISETC::BWTC(FR.num_v, FR.num_edge, FR.offsets, FR.csr);

        cudaEventRecord(t2);
        cudaEventSynchronize(t2);
        cudaEventElapsedTime(&elaps, t1, t2);
        elaps = elaps / REPS;
        std::cout << "Elapsed time: " << elaps << " ms with " << result << " triangles with Bitwise method." << std::endl;
    }

    return 0;
}