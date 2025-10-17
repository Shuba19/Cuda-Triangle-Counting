#ifndef CUDA_TRIANGLE_PROJ_v2
#define CUDA_TRIANGLE_PROJ_v2
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include "TriangleCounting.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <iostream>
int SearchTriangle(int num_v,int n_edges, std::vector<int>csr_size, std::vector<int> csr, int n_blocks);
#endif