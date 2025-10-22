#include "FileReader/FileReader.h"
#include <iostream>
#include "CUDA_Tri_v2/TriangleCounting.h"
#include "CUDA_Tri_v3/TensorCalculation.h"
#include <chrono>
#include <vector>
#include <cuda_runtime.h>

int main(int argc, char *argv[])
{
    MetisFR FR("./test/sample.graph");

    auto t_start = std::chrono::high_resolution_clock::now();

    int result = MatrixMulti(FR.num_v, FR.num_edge, FR.offsets, FR.csr, 128);


    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    std::cout << "MatrixMulti result: " << result << std::endl;
    std::cout << "Elapsed time: " << elapsed_ms << " ms" << std::endl;

    auto t1 = std::chrono::high_resolution_clock::now();
    result = SearchTriangle(FR.num_v,FR.num_edge,FR.offsets, FR.csr,128);

    auto t2 = std::chrono::high_resolution_clock::now();
    elapsed_ms =std::chrono::duration<double, std::milli>(t2 - t1).count();
    std::cout << "V2 result: " << result << std::endl;
    std::cout << "Elapsed time: " << elapsed_ms << " ms" << std::endl;
    return 0;
}