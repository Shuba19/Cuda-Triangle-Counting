#include "common_methods.h"

__device__ int triangular_col_from_id(int id)
{
    int col = 0;
    while ((col * (col + 1)) / 2 <= id)
        ++col;
    return col - 1;
}

__device__ int searchSourceNode(const int *ofs, int n, int id)
{
    int low = 0, high = n;
    while (low < high)
    {
        int mid = (low + high) >> 1;
        if (ofs[mid] <= id)
            low = mid + 1;
        else
            high = mid;
    }
    return low;
}

__device__ bool bin_search(int goal, int *v, int len)
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