#include "FileReader/FileReader.h"
#include <iostream>
#include "Libs/CUDA_Tri_Node_Iterator/NodeIterator.h"
#include "Libs/CUDA_Tri_Edge_Iterator/EdgeIterator.h"
#include <chrono>
#include <vector>
#include <cuda_runtime.h>

int main(int argc, char *argv[])
{
    MetisFR FR(argv[1]);
    double t_time = 0;
    auto t1 = std::chrono::high_resolution_clock::now();
    int result = SearchTriangle_Edge(FR.num_v, FR.num_edge, FR.offsets, FR.csr, 128, 5);

    
    auto t2 = std::chrono::high_resolution_clock::now();
    t_time += std::chrono::duration<double, std::milli>(t2 - t1).count();
    std::cout << "Elapsed time: " << t_time << " ms with "<<result<< " triangles." << std::endl;
    return 0;
}