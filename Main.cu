
#include "FileReader/FileReader.h"
#include <iostream>
#include "CUDA_Tri_v1/TriangleCounting.h"
#include "CUDA_Tri_v2/TriangleCounting.h"
#include <chrono>
#include <vector>

int main(int argc, char *argv[])
{
    MetisFR FR(argv[1]);
    /*
    auto t1 = std::chrono::high_resolution_clock::now();
    printCSR(FR.num_v, FR.num_edge, FR.CSR_size, FR.CSR);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Time v1 = " << std::chrono::duration<double>(t2 - t1).count() << "[s]" << std::endl;*/

    std::vector arr = {1024, 512,256, 128, 64, 32};
    for (auto i : arr)
    {
        auto t3 = std::chrono::high_resolution_clock::now();
        SearchTriangle(FR.num_v, FR.num_edge * 2, FR.offsets, FR.csr, i);
        auto t4 = std::chrono::high_resolution_clock::now();
        std::cout << "Time v2 = " << std::chrono::duration<double>(t4 - t3).count() << "[s] " << i << std::endl;
    }
    return 0;
}