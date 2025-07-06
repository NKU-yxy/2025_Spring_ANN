
// Version 1
// flat_scan_gpu.h

#pragma once
#include <vector>
#include <queue>
#include <cstdint>
#include <cuda_runtime.h>

// 只保留函数声明
std::priority_queue<std::pair<float, uint32_t>> flat_search_gpu(
    float* base, float* query, 
    size_t base_number, size_t vecdim, 
    size_t k);

// CUDA内核函数声明
void cuda_flat_search(const float* base, const float* query, 
                     size_t base_number, size_t vecdim, size_t k,
                     std::vector<std::pair<float, uint32_t>>& results);
    

// Version 2 
// flat_scan_gpu.h
/*
#pragma once
#include <queue>
#include <cstdint>

// 修改函数声明：第一个参数是查询向量，后三个是维度参数
std::priority_queue<std::pair<float, uint32_t>> flat_search_gpu(
    const float* query, size_t n, size_t dim, size_t k);

void init_gpu_memory(const float* base, size_t n, size_t dim);
void free_gpu_memory();
*/

// V4
/*
#pragma once
#include <cstddef>
#include <utility>
#include <vector>

// 保持原有接口但实现改为使用新的底层函数
std::priority_queue<std::pair<float, uint32_t>> flat_search_gpu(
    float* base, float* query, 
    size_t base_number, size_t vecdim, 
    size_t k);
*/

// V5.1
#pragma once
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

void gpu_search_interface(
    float* base, float* query, 
    size_t base_number, size_t vecdim, 
    float* distances, uint32_t* indices);

#ifdef __cplusplus
}
#endif