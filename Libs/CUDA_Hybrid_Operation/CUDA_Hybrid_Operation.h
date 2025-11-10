#ifndef CUDA_TRIANGLE_PROJ_v5
#define CUDA_TRIANGLE_PROJ_v5
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <stdio.h>
#include <cassert>
#include <mma.h>
#include <cuda_fp16.h>
namespace HTC
{
    int64_t H_triangle_counting(int num_v, int n_edges, std::vector<int> &offsets, std::vector<int> &csr);
}
#endif