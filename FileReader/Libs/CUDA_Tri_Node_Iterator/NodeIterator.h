#ifndef CUDA_TRIANGLE_PROJ_v2
#define CUDA_TRIANGLE_PROJ_v2

#include "../CommonMethods/common_methods.h"
out_type SearchTriangle_Node_Iterator(int num_v,int n_edges, std::vector<int>& csr_size, std::vector<int>& csr, bool undirect);
#endif