#ifndef CUDA_TRIANGLE_PROJ_v3
#define CUDA_TRIANGLE_PROJ_v3

#include "../CommonMethods/common_methods.h"
out_type TTC(int num_v,int n_edges, std::vector<int>offsets, std::vector<int> csr);
out_type TTC_v2(int num_v, int n_edges, const std::vector<int> &offsets, const std::vector<int> &csr);
#endif