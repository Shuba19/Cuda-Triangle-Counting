#ifndef CUDA_TRIANGLE_PROJ_v1
#define CUDA_TRIANGLE_PROJ_v1
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <numeric>
int SearchTriangle_Edge(int num_v,int n_edges, std::vector<int>& offsets, std::vector<int>& csr, int n_th);
#endif