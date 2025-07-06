
// flat_scan_gpu.cu (简化版)
// Version 1
/*
#include "flat_scan_gpu.h"
#include <cuda_runtime.h>

__global__ void compute_distances(const float* base, const float* query,
                                size_t base_number, size_t vecdim,
                                float* distances, uint32_t* indices) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= base_number) return;

    float dis = 0.0f;
    for (size_t d = 0; d < vecdim; ++d) {
        dis += base[i * vecdim + d] * query[d];
    }
    distances[i] = 1.0f - dis;
    indices[i] = i;
}

void cuda_flat_search(const float* base, const float* query, 
                     size_t base_number, size_t vecdim, size_t k,
                     std::vector<std::pair<float, uint32_t>>& results) {
    // 分配设备内存
    float *d_base, *d_query, *d_distances;
    uint32_t *d_indices;
    
    cudaMalloc(&d_base, base_number * vecdim * sizeof(float));
    cudaMalloc(&d_query, vecdim * sizeof(float));
    cudaMalloc(&d_distances, base_number * sizeof(float));
    cudaMalloc(&d_indices, base_number * sizeof(uint32_t));
    
    // 拷贝数据到设备
    cudaMemcpy(d_base, base, base_number * vecdim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_query, query, vecdim * sizeof(float), cudaMemcpyHostToDevice);

    // 启动内核
    const size_t block_size = 256;
    const size_t grid_size = (base_number + block_size - 1) / block_size;
    
    compute_distances<<<grid_size, block_size>>>(d_base, d_query, base_number, vecdim, d_distances, d_indices);
    cudaDeviceSynchronize();
    
    // 拷贝结果回主机
    float* h_distances = new float[base_number];
    uint32_t* h_indices = new uint32_t[base_number];
    
    cudaMemcpy(h_distances, d_distances, base_number * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_indices, d_indices, base_number * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    
    // 填充结果
    results.resize(base_number);
    for(size_t i = 0; i < base_number; ++i) {
        results[i] = {h_distances[i], h_indices[i]};
    }
    
    // 释放内存
    delete[] h_distances;
    delete[] h_indices;
    cudaFree(d_base);
    cudaFree(d_query);
    cudaFree(d_distances);
    cudaFree(d_indices);
}

std::priority_queue<std::pair<float, uint32_t>> flat_search_gpu(
    float* base, float* query, 
    size_t base_number, size_t vecdim, 
    size_t k) {
    std::vector<std::pair<float, uint32_t>> temp_results(base_number);
    
    cuda_flat_search(base, query, base_number, vecdim, k, temp_results);
    
    std::priority_queue<std::pair<float, uint32_t>> q;
    for (size_t i = 0; i < base_number; ++i) {
        if (q.size() < k) {
            q.push(temp_results[i]);
        } else if (temp_results[i].first < q.top().first) {
            q.pop();
            q.push(temp_results[i]);
        }
    }
    return q;
}
  */


// Version 2
/*
#include "flat_scan_gpu.h"
#include <cfloat>
#include <queue>

// 使用结构体包装指针避免类型混淆
struct DevicePointers {
    float* base;
    float* query;
    float* distances;
    int* indices;  // 保持为int类型
    size_t dim = 0;
};

static DevicePointers dev_ptrs;

void init_gpu_memory(const float* base, size_t n, size_t dim) {
    if (dev_ptrs.base && dim == dev_ptrs.dim) return;
    
    // 释放旧内存
    if (dev_ptrs.base) {
        cudaFree(dev_ptrs.base);
        cudaFree(dev_ptrs.query);
        cudaFree(dev_ptrs.distances);
        cudaFree(dev_ptrs.indices);
    }

    // 分配新内存
    cudaMalloc((void**)&dev_ptrs.base, n * dim * sizeof(float));
    cudaMalloc((void**)&dev_ptrs.query, dim * sizeof(float));
    cudaMalloc((void**)&dev_ptrs.distances, n * sizeof(float));
    cudaMalloc((void**)&dev_ptrs.indices, n * sizeof(int));
    
    cudaMemcpy(dev_ptrs.base, base, n * dim * sizeof(float), cudaMemcpyHostToDevice);
    dev_ptrs.dim = dim;
}

void free_gpu_memory() {
    if (dev_ptrs.base) {
        cudaFree(dev_ptrs.base);
        cudaFree(dev_ptrs.query);
        cudaFree(dev_ptrs.distances);
        cudaFree(dev_ptrs.indices);
        dev_ptrs = DevicePointers();  // 重置结构体
    }
}

// 修改后的内核函数（显式指定指针类型）
__global__ void compute_distances_kernel(
    const float* __restrict__ base,
    const float* __restrict__ query,
    float* __restrict__ distances,
    int* __restrict__ indices,
    size_t n, size_t dim) {
    
    extern __shared__ float s_query[];
    int tid = threadIdx.x;
    
    for (int i = tid; i < dim; i += blockDim.x) {
        s_query[i] = query[i];
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + tid;
    if (idx >= n) return;

    float dis = 0.0f;
    for (int d = 0; d < dim; ++d) {
        dis += base[idx * dim + d] * s_query[d];
    }
    distances[idx] = 1.0f - dis;
    indices[idx] = idx;
}

// 简化版Top-K选择（无需排序）
__global__ void select_top_k(
    const float* __restrict__ distances,
    const int* __restrict__ indices,
    float* __restrict__ top_dists,
    int* __restrict__ top_indices,
    size_t n, size_t k) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= k) return;

    float min_val = FLT_MAX;
    int min_idx = -1;
    
    for (size_t i = idx; i < n; i += k) {
        if (distances[i] < min_val) {
            min_val = distances[i];
            min_idx = indices[i];
        }
    }
    
    top_dists[idx] = min_val;
    top_indices[idx] = min_idx;
}

std::priority_queue<std::pair<float, uint32_t>> flat_search_gpu(
    const float* query, size_t n, size_t dim, size_t k) {
    
    // 拷贝查询向量
    cudaMemcpy(dev_ptrs.query, query, dim * sizeof(float), cudaMemcpyHostToDevice);

    // 计算距离
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    compute_distances_kernel<<<grid, block, dim*sizeof(float)>>>(
        dev_ptrs.base, dev_ptrs.query, 
        dev_ptrs.distances, dev_ptrs.indices, 
        n, dim);

    // 分配临时存储
    float* d_top_dists;
    int* d_top_indices;
    cudaMalloc(&d_top_dists, k * sizeof(float));
    cudaMalloc(&d_top_indices, k * sizeof(int));

    // 选择Top-K
    select_top_k<<<(k + 255)/256, 256>>>(
        dev_ptrs.distances, dev_ptrs.indices,
        d_top_dists, d_top_indices,
        n, k);

    // 取回结果
    std::vector<float> h_dists(k);
    std::vector<int> h_indices(k);
    cudaMemcpy(h_dists.data(), d_top_dists, k*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_indices.data(), d_top_indices, k*sizeof(int), cudaMemcpyDeviceToHost);

    // 构建优先队列
    std::priority_queue<std::pair<float, uint32_t>> pq;
    for (size_t i = 0; i < k; ++i) {
        pq.emplace(h_dists[i], static_cast<uint32_t>(h_indices[i]));
    }

    // 释放临时内存
    cudaFree(d_top_dists);
    cudaFree(d_top_indices);
    
    return pq;
}
    */

// Version 3
/*
#include "flat_scan_gpu.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <iostream>
#include <cfloat>


#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
}

// 使用共享内存优化距离计算
__global__ void compute_distances_optimized(const float* __restrict__ base, 
                                          const float* __restrict__ query,
                                          float* __restrict__ distances,
                                          uint32_t* __restrict__ indices,
                                          size_t base_number, 
                                          size_t vecdim) {
    extern __shared__ float shared_query[];
    
    // 每个线程加载一部分query到共享内存
    int tid = threadIdx.x;
    for (int i = tid; i < vecdim; i += blockDim.x) {
        shared_query[i] = query[i];
    }
    __syncthreads();

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= base_number) return;

    float dis = 0.0f;
    #pragma unroll(4)
    for (size_t d = 0; d < vecdim; ++d) {
        dis += base[idx * vecdim + d] * shared_query[d];
    }
    
    // 使用1.0f - dis作为距离度量(保持与原始代码一致)
    distances[idx] = 1.0f - dis;
    indices[idx] = idx;
}

// 简单的Top-K选择内核
__global__ void select_top_k(const float* distances, 
                            const uint32_t* indices,
                            float* top_dists,
                            uint32_t* top_indices,
                            size_t n, size_t k) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= k) return;

    float min_val = FLT_MAX;
    uint32_t min_idx = 0;
    
    // 每个线程处理n/k个元素
    for (size_t i = idx; i < n; i += k) {
        if (distances[i] < min_val) {
            min_val = distances[i];
            min_idx = indices[i];
        }
    }
    
    top_dists[idx] = min_val;
    top_indices[idx] = min_idx;
}

// 优化后的GPU搜索实现
void cuda_flat_search_optimized(const float* base, const float* query, 
                              size_t base_number, size_t vecdim, size_t k,
                              std::vector<std::pair<float, uint32_t>>& results) {
    // 分配设备内存
    float *d_base, *d_query, *d_distances;
    uint32_t *d_indices;
    float *d_top_dists;
    uint32_t *d_top_indices;
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_base, base_number * vecdim * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_query, vecdim * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_distances, base_number * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_indices, base_number * sizeof(uint32_t)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_top_dists, k * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_top_indices, k * sizeof(uint32_t)));

    // 创建CUDA流
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
    
    // 异步拷贝数据到设备
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_base, base, base_number * vecdim * sizeof(float), 
                                   cudaMemcpyHostToDevice, stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_query, query, vecdim * sizeof(float), 
                                   cudaMemcpyHostToDevice, stream));
    
    // 计算最佳block大小(基于T4的特性)
    int block_size = 256;
    if (vecdim <= 64) block_size = 128;
    if (vecdim <= 32) block_size = 64;
    
    size_t grid_size = (base_number + block_size - 1) / block_size;
    
    // 启动优化后的内核
    compute_distances_optimized<<<grid_size, block_size, vecdim * sizeof(float), stream>>>(
        d_base, d_query, d_distances, d_indices, base_number, vecdim);
    
    // 选择Top-K
    dim3 topk_block(256);
    dim3 topk_grid((k + topk_block.x - 1) / topk_block.x);
    select_top_k<<<topk_grid, topk_block, 0, stream>>>(
        d_distances, d_indices, d_top_dists, d_top_indices, base_number, k);
    
    // 分配主机内存并拷贝结果
    std::vector<float> h_top_dists(k);
    std::vector<uint32_t> h_top_indices(k);
    
    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_top_dists.data(), d_top_dists, k * sizeof(float), 
                    cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_top_indices.data(), d_top_indices, k * sizeof(uint32_t), 
                    cudaMemcpyDeviceToHost, stream));
    
    // 同步流
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    
    // 填充结果
    results.resize(k);
    for (size_t i = 0; i < k; ++i) {
        results[i] = {h_top_dists[i], h_top_indices[i]};
    }
    
    // 释放资源
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
    CHECK_CUDA_ERROR(cudaFree(d_base));
    CHECK_CUDA_ERROR(cudaFree(d_query));
    CHECK_CUDA_ERROR(cudaFree(d_distances));
    CHECK_CUDA_ERROR(cudaFree(d_indices));
    CHECK_CUDA_ERROR(cudaFree(d_top_dists));
    CHECK_CUDA_ERROR(cudaFree(d_top_indices));
}

// 优化后的接口函数
std::priority_queue<std::pair<float, uint32_t>> flat_search_gpu(
    float* base, float* query, 
    size_t base_number, size_t vecdim, 
    size_t k) {
    std::vector<std::pair<float, uint32_t>> temp_results(k);
    
    cuda_flat_search_optimized(base, query, base_number, vecdim, k, temp_results);
    
    std::priority_queue<std::pair<float, uint32_t>> q;
    for (size_t i = 0; i < k; ++i) {
        q.push(temp_results[i]);
    }
    return q;
}
    */

// V4
/*
#include "flat_scan_gpu.h"
#include <cuda_runtime.h>
#include <cfloat>
#include <cstdio>
#include <cstdlib>

// 移除所有模板和STL复杂用法
#define CHECK_CUDA(call) {\
    cudaError_t err = call;\
    if (err != cudaSuccess) {\
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s)\n", __FILE__, __LINE__, err, cudaGetErrorString(err));\
        exit(1);\
    }\
}

__global__ void compute_distances(const float* base, const float* query,
                                size_t base_number, size_t vecdim,
                                float* distances, uint32_t* indices) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= base_number) return;

    float dis = 0.0f;
    for (size_t d = 0; d < vecdim; ++d) {
        dis += base[i * vecdim + d] * query[d];
    }
    distances[i] = 1.0f - dis;
    indices[i] = i;
}

extern "C" void cuda_flat_search_simple(
    const float* base, const float* query, 
    size_t base_number, size_t vecdim, size_t k,
    float* distances, uint32_t* indices) {
    
    float *d_base, *d_query, *d_distances;
    uint32_t *d_indices;
    
    CHECK_CUDA(cudaMalloc(&d_base, base_number * vecdim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_query, vecdim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_distances, base_number * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_indices, base_number * sizeof(uint32_t)));
    
    CHECK_CUDA(cudaMemcpy(d_base, base, base_number * vecdim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_query, query, vecdim * sizeof(float), cudaMemcpyHostToDevice));

    const size_t block_size = 256;
    const size_t grid_size = (base_number + block_size - 1) / block_size;
    
    compute_distances<<<grid_size, block_size>>>(d_base, d_query, base_number, vecdim, d_distances, d_indices);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    CHECK_CUDA(cudaMemcpy(distances, d_distances, base_number * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(indices, d_indices, base_number * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    
    CHECK_CUDA(cudaFree(d_base));
    CHECK_CUDA(cudaFree(d_query));
    CHECK_CUDA(cudaFree(d_distances));
    CHECK_CUDA(cudaFree(d_indices));
}

// #pragma once
#include <cstddef>
#include <utility>
#include <vector>

// 保持原有接口但实现改为使用新的底层函数
std::priority_queue<std::pair<float, uint32_t>> flat_search_gpu(
    float* base, float* query, 
    size_t base_number, size_t vecdim, 
    size_t k);

*/


// V4.1
/*
#include "flat_scan_gpu.h"
#include <cuda_runtime.h>
#include <queue>
#include <vector>
#include <cfloat>
#include <cstdio>
#include <cstdlib>

#define CHECK_CUDA(call) \
do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d code=%d\n", __FILE__, __LINE__, err); \
        exit(1); \
    } \
} while (0)

__global__ void compute_distances(const float* base, const float* query,
                                size_t base_number, size_t vecdim,
                                float* distances, uint32_t* indices) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= base_number) return;

    float dis = 0.0f;
    for (size_t d = 0; d < vecdim; ++d) {
        dis += base[i * vecdim + d] * query[d];
    }
    distances[i] = 1.0f - dis;
    indices[i] = i;
}

std::priority_queue<std::pair<float, uint32_t>> flat_search_gpu(
    float* base, float* query, 
    size_t base_number, size_t vecdim, 
    size_t k) {
    
    // 分配设备内存
    float *d_base, *d_query, *d_distances;
    uint32_t *d_indices;
    
    CHECK_CUDA(cudaMalloc(&d_base, base_number * vecdim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_query, vecdim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_distances, base_number * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_indices, base_number * sizeof(uint32_t)));
    
    // 拷贝数据到设备
    CHECK_CUDA(cudaMemcpy(d_base, base, base_number * vecdim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_query, query, vecdim * sizeof(float), cudaMemcpyHostToDevice));

    // 计算block和grid大小
    const size_t block_size = 256;
    const size_t grid_size = (base_number + block_size - 1) / block_size;
    
    // 启动内核
    compute_distances<<<grid_size, block_size>>>(d_base, d_query, base_number, vecdim, d_distances, d_indices);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // 拷贝结果回主机
    std::vector<float> distances(base_number);
    std::vector<uint32_t> indices(base_number);
    CHECK_CUDA(cudaMemcpy(distances.data(), d_distances, base_number * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(indices.data(), d_indices, base_number * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    
    // 构建优先队列
    std::priority_queue<std::pair<float, uint32_t>> q;
    for (size_t i = 0; i < base_number; ++i) {
        if (q.size() < k) {
            q.push({distances[i], indices[i]});
        } else if (distances[i] < q.top().first) {
            q.pop();
            q.push({distances[i], indices[i]});
        }
    }
    
    // 释放内存
    CHECK_CUDA(cudaFree(d_base));
    CHECK_CUDA(cudaFree(d_query));
    CHECK_CUDA(cudaFree(d_distances));
    CHECK_CUDA(cudaFree(d_indices));
    
    return q;
} 
*/

// V5
/*
#include "flat_scan_gpu.h"
#include <cuda_runtime.h>
#include <queue>
#include <vector>
#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <cstdlib>

#define CHECK_CUDA(call) {\
    cudaError_t err = call;\
    if (err != cudaSuccess) {\
        printf("CUDA error at %s:%d code=%d\n", __FILE__, __LINE__, err);\
        exit(1);\
    }}

// 优化1：使用共享内存和向量化加载
__global__ void compute_distances_opt(const float* __restrict__ base, 
                                    const float* __restrict__ query,
                                    float* __restrict__ distances,
                                    uint32_t* __restrict__ indices,
                                    size_t base_number, 
                                    size_t vecdim) {
    extern __shared__ float shared_query[];
    
    // 协作加载query到共享内存
    int tid = threadIdx.x;
    for (int i = tid; i < vecdim; i += blockDim.x) {
        shared_query[i] = query[i];
    }
    __syncthreads();

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= base_number) return;

    // 使用向量化计算
    float dis = 0.0f;
    const float* base_ptr = base + idx * vecdim;
    #pragma unroll(4)
    for (size_t d = 0; d < vecdim; ++d) {
        dis += base_ptr[d] * shared_query[d];
    }
    
    distances[idx] = 1.0f - dis;  // 保持与原始算法一致
    indices[idx] = idx;
}

// 优化2：使用CUDA流和异步操作
std::priority_queue<std::pair<float, uint32_t>> flat_search_gpu(
    float* base, float* query, 
    size_t base_number, size_t vecdim, 
    size_t k) {
    
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    
    // 分配固定内存(pinned memory)加速传输
    float *pinned_dist, *d_base, *d_query, *d_dist;
    uint32_t *pinned_idx, *d_idx;
    
    CHECK_CUDA(cudaMallocHost(&pinned_dist, base_number * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&pinned_idx, base_number * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_base, base_number * vecdim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_query, vecdim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dist, base_number * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_idx, base_number * sizeof(uint32_t)));

    // 异步拷贝
    CHECK_CUDA(cudaMemcpyAsync(d_base, base, base_number * vecdim * sizeof(float), 
                             cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_query, query, vecdim * sizeof(float), 
                             cudaMemcpyHostToDevice, stream));

    // 根据维度自动选择block大小
    int block_size = (vecdim <= 64) ? 128 : 256;
    size_t grid_size = (base_number + block_size - 1) / block_size;
    size_t shared_mem = vecdim * sizeof(float);
    
    compute_distances_opt<<<grid_size, block_size, shared_mem, stream>>>(
        d_base, d_query, d_dist, d_idx, base_number, vecdim);

    // 异步拷贝回
    CHECK_CUDA(cudaMemcpyAsync(pinned_dist, d_dist, base_number * sizeof(float), 
                             cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaMemcpyAsync(pinned_idx, d_idx, base_number * sizeof(uint32_t), 
                             cudaMemcpyDeviceToHost, stream));
    
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // 使用partial_sort代替优先队列(更快)
    std::vector<std::pair<float, uint32_t>> results(base_number);
    for (size_t i = 0; i < base_number; ++i) {
        results[i] = {pinned_dist[i], pinned_idx[i]};
    }
    
    std::partial_sort(results.begin(), results.begin() + k, results.end());
    std::priority_queue<std::pair<float, uint32_t>> q;
    for (size_t i = 0; i < k; ++i) {
        q.push(results[i]);
    }

    // 释放资源
    CHECK_CUDA(cudaFreeHost(pinned_dist));
    CHECK_CUDA(cudaFreeHost(pinned_idx));
    CHECK_CUDA(cudaFree(d_base));
    CHECK_CUDA(cudaFree(d_query));
    CHECK_CUDA(cudaFree(d_dist));
    CHECK_CUDA(cudaFree(d_idx));
    CHECK_CUDA(cudaStreamDestroy(stream));
    
    return q;
}
    */

// V5.1
#include "flat_scan_gpu.h"
#include <cuda_runtime.h>

#define CHECK_CUDA(call) \
do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d code=%d\n", __FILE__, __LINE__, err); \
        exit(1); \
    } \
} while (0)

__global__ void compute_distances_opt(const float* __restrict__ base, 
                                    const float* __restrict__ query,
                                    float* __restrict__ distances,
                                    uint32_t* __restrict__ indices,
                                    size_t base_number, 
                                    size_t vecdim) {
    extern __shared__ float shared_query[];
    
    int tid = threadIdx.x;
    for (int i = tid; i < vecdim; i += blockDim.x) {
        shared_query[i] = query[i];
    }
    __syncthreads();

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= base_number) return;

    float dis = 0.0f;
    const float* base_ptr = base + idx * vecdim;
    #pragma unroll(4)
    for (size_t d = 0; d < vecdim; ++d) {
        dis += base_ptr[d] * shared_query[d];
    }
    
    distances[idx] = 1.0f - dis;
    indices[idx] = idx;
}

void gpu_search_interface(
    float* base, float* query, 
    size_t base_number, size_t vecdim, 
    float* distances, uint32_t* indices) {
    
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    
    float *d_base, *d_query, *d_dist;
    uint32_t *d_idx;
    
    CHECK_CUDA(cudaMalloc(&d_base, base_number * vecdim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_query, vecdim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dist, base_number * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_idx, base_number * sizeof(uint32_t)));

    CHECK_CUDA(cudaMemcpyAsync(d_base, base, base_number * vecdim * sizeof(float), 
                             cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_query, query, vecdim * sizeof(float), 
                             cudaMemcpyHostToDevice, stream));

    int block_size = (vecdim <= 64) ? 128 : 256;
    size_t grid_size = (base_number + block_size - 1) / block_size;
    
    compute_distances_opt<<<grid_size, block_size, vecdim*sizeof(float), stream>>>(
        d_base, d_query, d_dist, d_idx, base_number, vecdim);

    CHECK_CUDA(cudaMemcpyAsync(distances, d_dist, base_number * sizeof(float), 
                             cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaMemcpyAsync(indices, d_idx, base_number * sizeof(uint32_t), 
                             cudaMemcpyDeviceToHost, stream));
    
    CHECK_CUDA(cudaStreamSynchronize(stream));
    
    CHECK_CUDA(cudaFree(d_base));
    CHECK_CUDA(cudaFree(d_query));
    CHECK_CUDA(cudaFree(d_dist));
    CHECK_CUDA(cudaFree(d_idx));
    CHECK_CUDA(cudaStreamDestroy(stream));
}