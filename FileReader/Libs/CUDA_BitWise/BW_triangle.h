#ifndef CUDA_TRIANGLE_PROJ_BWISE
#define CUDA_TRIANGLE_PROJ_BWISE
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <stdio.h>
#include <algorithm>
#include <cmath>
#include <mma.h>
#include <cuda_fp16.h>
namespace BWISETC
{
    int64_t BWTC(int num_v, int n_edges, std::vector<int> offsets, std::vector<int> csr);
}
#endif