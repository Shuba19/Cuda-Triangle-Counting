#ifndef CUDA_TRIANGLE_PROJ_v1
#define CUDA_TRIANGLE_PROJ_v1
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <numeric>
int64_t SearchTriangle_Edge_Iterator(int num_v,int n_edges, std::vector<int>& offsets, std::vector<int>& csr, bool undirect);
#endif