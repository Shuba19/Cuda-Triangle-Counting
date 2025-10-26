#ifndef CUDA_TRIANGLE_PROJ_v3
#define CUDA_TRIANGLE_PROJ_v3
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <stdio.h>
#include <cassert>
#include <mma.h>
#include <cuda_fp16.h>
int TTC(int num_v,int n_edges, std::vector<int>offsets, std::vector<int> csr);
#endif