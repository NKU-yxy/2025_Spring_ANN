// flat_search_sq.h

#pragma once
#include <queue>
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <random>

// find 数据集内max abs（ 即max(最大正数,最小负数) ），拿来确认映射区间的上下限。
// 所有float都要写成类似0.0f的样子，不然会自动变成64位的double
inline float find_max_abs(const float* data, size_t num, size_t dim) 
{
    // 存储最后找到的最大的MAX ABS
    float max_val = 0.0f;
    // 遍历所有float
    for (size_t i = 0; i < num * dim; i++) 
    {
        // 取当前float的abs 与已有的最大abs进行比较
        max_val = std::max(max_val, std::abs(data[i]));
    }
    return max_val;
}

// float → uint8 转换
inline uint8_t change_float_to_uint(float val, float max_abs) 
{
    // 为了防止数值越界，这一步是必要的，将任意输入都限制在 [-max_abs,max_abs] 内
    // 若val > max_abs 那就直接转化为max_abs 
    // 若val < -max_abs 那就直接转换为-max_abs
    float changed_val = std::max(-max_abs, std::min(val, max_abs));

    // 线性映射：实现转换
    return static_cast<uint8_t>((changed_val + max_abs) * 255.0f / (2 * max_abs));
}

// 逆转换：uint8 → float
inline void change_uint8_to_float(const uint8_t* uint8_data, float* float_data, size_t num, size_t dim, float max_abs) 
{
    for (size_t i = 0; i < num * dim; i++) 
    {
        // 归一化到 [0, 1]
        float normalized = static_cast<float>(uint8_data[i]) / 255.0f; 
        // 还原到原范围 [-max_abs, max_abs]
        float_data[i] = normalized * 2.0f * max_abs - max_abs;        
    }
}

// 将整个dataset都从float -> uint8
inline void change_dataset(const float* float_data, uint8_t* uint8_data, size_t num, size_t dim, float max_abs) 
{
    // 二重循环遍历data，全部实现量化
    for (size_t i = 0; i < num; i++) 
    {
        for (size_t j = 0; j < dim; j++) 
        {
            uint8_data[i * dim + j] = change_float_to_uint(float_data[i * dim + j], max_abs);
        }
    }
}

// uint8类型的内积计算
inline uint32_t dot_product_u8(const uint8_t* a, const uint8_t* b, size_t dim)
 {
    // 8bits * 8bits 最大可以是16bits 此处用32bits存储
    uint32_t result = 0;
    // 遍历所有维度，每个维度相乘后求和累加即可
    for (size_t i = 0; i < dim; i++) 
    {
        result += static_cast<uint32_t>(a[i]) * static_cast<uint32_t>(b[i]);
    }
    return result;
}

// 返回类型为一个min heap
inline std::priority_queue<std::pair<float, uint32_t>, 
                          std::vector<std::pair<float, uint32_t>>,
                          std::greater<std::pair<float, uint32_t>>>
flat_search_sq(const uint8_t* base_q, const uint8_t* query_q,
               size_t base_number, size_t vecdim, size_t k, float max_abs) 
{
    // uint8 ---> float 的因子为float
    const float scale = (2.0f * max_abs / 255.0f);
    // 对于内积的放大，应该采用因子的平方
    const float scale2 = scale * scale;

    // 计算 query 向量的平方模长
    float query_norm = std::sqrt(dot_product_u8(query_q, query_q, vecdim));

    std::priority_queue<std::pair<float, uint32_t>,
                        std::vector<std::pair<float, uint32_t>>,
                        std::greater<std::pair<float, uint32_t>>> pq;

    for (size_t i = 0; i < base_number; i++) 
    {
        const uint8_t* vec = base_q + i * vecdim;
        
        uint32_t raw_ip = dot_product_u8(vec, query_q, vecdim);
        // 内积乘上平方的放大因子得到float类型
        float ip = raw_ip * scale2;
        
        // 计算当前vector的模长
        float base_norm = std::sqrt(dot_product_u8(vec, vec, vecdim)) * scale;
        // 计算Cosine相似度，分母加上1e-6防止出现除以0的情况
        float cosine_sim = ip / (query_norm * base_norm + 1e-6f); 
        
        // vector插入最小堆的逻辑同前
        if (pq.size() < k) 
        {
            pq.emplace(cosine_sim, i);
        } 
        else if (cosine_sim > pq.top().first) 
        {
            pq.pop();
            pq.emplace(cosine_sim, i);
        }
    }

    return pq;
}
