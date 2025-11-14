#ifndef CUDA_TRIANGLE_PROJ_BWISE
#define CUDA_TRIANGLE_PROJ_BWISE
#include "../CommonMethods/common_methods.h"
namespace BWISETC
{
    out_type BWTC(int num_v, int n_edges, std::vector<int> offsets, std::vector<int> csr);
}
#endif