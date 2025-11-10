#include "CUDA_Hybrid_Operation.h"
#define THOLD 5
#define CHECK(call)                                                         \
    {                                                                       \
        const cudaError_t error = call;                                     \
        if (error != cudaSuccess)                                           \
        {                                                                   \
            printf("Error %s : %d\n", __FILE__, __LINE__);                  \
            printf("code:%d, reason:%s", error, cudaGetErrorString(error)); \
            exit(1);                                                        \
        }                                                                   \
    }
namespace HTC
{
    __device__ bool static bin_search(int goal, int *v, int len)
    {
        int l = 0;
        int h = len;
        while (l < h)
        {
            int mid = l + ((h - l) >> 1);
            int v_mid = v[mid];
            if (v_mid < goal)
            {
                l = mid + 1;
            }
            else
            {
                h = mid;
            }
        }
        return (l < len) && (v[l] == goal);
    }

    __device__ __forceinline__ int searchSourceNode(const int *arr, int n, int value)
    {
        int low = 0, high = n;
        while (low < high)
        {
            int mid = (low + high) >> 1;
            if (arr[mid] <= value)
                low = mid + 1;
            else
                high = mid;
        }
        return low;
    }
    // Edge-iterator: process only edges (u, v) where u is in assigned_id and v > u
    // Count triangles (u, v, w) with ordering u < v < w via two-pointer intersection.
    __global__ void partialEdgeIterator(int p_id,
                                        const int *assigned_id,
                                        const int *ofs,
                                        const int *csr,
                                        int *results)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= p_id) return;

        int u = assigned_id[tid];
        int u_begin = ofs[u];
        int u_end   = ofs[u + 1];

        int tri = 0;

        // For each neighbor v of u with v > u
        for (int ei = u_begin; ei < u_end; ++ei)
        {
            int v = csr[ei];
            if (v <= u) continue;

            int i = u_begin;
            int i_end = u_end;
            int j = ofs[v];
            int j_end = ofs[v + 1];

            // Merge-based intersection; count only w > v to enforce u < v < w
            while (i < i_end && j < j_end)
            {
                int a = csr[i];
                int b = csr[j];
                if (a == b)
                {
                    if (a > v) ++tri;
                    ++i; ++j;
                }
                else if (a < b)
                {
                    ++i;
                }
                else
                {
                    ++j;
                }
            }
        }

        results[u] = tri;
    }

    __global__ void static partialNodeIterator(int p_id, int *assigned_id, int *ofs, int *csr, int *results)
    {
        int id_th = blockIdx.x * blockDim.x + threadIdx.x;
        if (id_th < p_id)
        {
            int id = assigned_id[id_th];
            int of1, of2;
            of1 = ofs[id];
            of2 = ofs[id + 1];
            int count = 0;
            #pragma unroll
            for (int i = of1; i < of2; i++)
            {
                int index = csr[i];
                if (index <= id)
                    continue;
                for (int j = ofs[index]; j < ofs[index + 1]; j++)
                {
                    int pivot = csr[j];
                    if (pivot <= index)
                        continue;
                    count += bin_search(pivot, &csr[of1], of2 - of1) ? 1 : 0;
                }
            }
            results[id] = count;
        }
    }

    int64_t H_triangle_counting(int num_v, int n_edges, std::vector<int> &offsets, std::vector<int> &csr)
    {
        cudaSetDevice(0);
        n_edges = n_edges << 1;
        int *d_csr, *d_ofs, *d_res;
        int *d_list_ni, *d_list_ei;
        std::vector<int> list_ei, list_ni;
        for (int i = 1; i < offsets.size(); i++)
        {
            int size = offsets[i] - offsets[i - 1];
            if (size < THOLD)
                list_ni.push_back(i - 1);
            else
                list_ei.push_back(i - 1);
        }

        d_csr = nullptr;
        d_ofs = nullptr;
        d_res = nullptr;
        CHECK(cudaMalloc(&d_ofs, (offsets.size()) * sizeof(int)));
        CHECK(cudaMalloc(&d_csr, n_edges * sizeof(int)));
        CHECK(cudaMalloc(&d_res, num_v * sizeof(int)));
        CHECK(cudaMalloc(&d_list_ni, list_ni.size() * sizeof(int)));
        CHECK(cudaMalloc(&d_list_ei, list_ei.size() * sizeof(int)));

        CHECK(cudaMemcpyAsync(d_csr, csr.data(), csr.size() * sizeof(int), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpyAsync(d_ofs, offsets.data(), offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
        CHECK(cudaMemsetAsync(d_res, 0, num_v * sizeof(int)));
        CHECK(cudaMemcpyAsync(d_list_ni, list_ni.data(), list_ni.size() * sizeof(int), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpyAsync(d_list_ei, list_ei.data(), list_ei.size() * sizeof(int), cudaMemcpyHostToDevice));
        //std::cout << "NI : " << list_ni.size() << "     NE : " << list_ei.size() << std::endl;
        if (!list_ni.empty()) {
            dim3 nI_block(128);
            dim3 nI_grid((static_cast<unsigned>(list_ni.size()) + nI_block.x - 1) / nI_block.x);
            partialNodeIterator<<<nI_grid, nI_block>>>(static_cast<int>(list_ni.size()), d_list_ni, d_ofs, d_csr, d_res);
            CHECK(cudaGetLastError());
        }


        if (!list_ei.empty()) {
            dim3 eI_block(128);
            dim3 eI_grid((static_cast<unsigned>(list_ei.size()) + eI_block.x - 1) / eI_block.x);
            partialEdgeIterator<<<eI_grid, eI_block>>>(static_cast<int>(list_ei.size()), d_list_ei, d_ofs, d_csr, d_res);
            CHECK(cudaGetLastError());
        }

        CHECK(cudaDeviceSynchronize());
        std::vector<int> results(num_v);
        cudaMemcpy(results.data(), d_res, num_v * sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(d_res);
        cudaFree(d_csr);
        cudaFree(d_ofs);
        cudaFree(d_list_ei);
        cudaFree(d_list_ni);
        int64_t n_tri = 0;
        for(auto i : results)
            n_tri += i;
        return n_tri;
    }
}