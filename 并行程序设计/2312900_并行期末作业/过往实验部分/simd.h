
// 2.2
// recall: 0.872   time: 5300
// simd.h
#ifndef SIMD_H
#define SIMD_H

#include <cstdint>
#include <vector>
#include <algorithm>
#include <queue>
#include <arm_neon.h>

// 定义两种优先队列类型
template<typename T1, typename T2>
using MinQueue = std::priority_queue<
    std::pair<T1, T2>,
    std::vector<std::pair<T1, T2>>,
    std::greater<std::pair<T1, T2>>
>;

template<typename T1, typename T2>
using MaxQueue = std::priority_queue<
    std::pair<T1, T2>,
    std::vector<std::pair<T1, T2>>,
    std::less<std::pair<T1, T2>>
>;

inline void simd_quantize_global(
    const float* src, uint8_t* dst,
    size_t n, size_t dim,
    float global_max_abs,
    std::vector<float>& mean
) 
{
    mean.resize(dim, 0.0f);
    
    // 阶段1：计算各维度均值
    #pragma omp parallel for
    for (size_t j = 0; j < dim; ++j) {
        float sum = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            sum += src[i * dim + j];
        }
        mean[j] = sum / n;
    }
    
    // 阶段2：中心化+量化
    const float scale = 127.0f / global_max_abs;
    #pragma omp parallel for
    for (size_t i = 0; i < n * dim; ++i) {
        float val = src[i] - mean[i % dim];
        // 代替clamp，处理数据越界
        size_t j = i % dim;
        if(val < -global_max_abs)
        {
            val = -global_max_abs;
        }
        else if(val > global_max_abs)
        {
            val = global_max_abs;
        }

        
        dst[i] = static_cast<uint8_t>(val * scale + 128.0f);
    }
}

inline float exact_dot_product(const float* a, const float* b, size_t dim)
{
    /*
    此处以dim=8为例子:
      设a=[a1~a8] b=[b1~b8]
      init: sum=[0,0,0,0]
      第一次循环:
          va=[a1~a4]
          vb=[b1~b4]
          sum=[a1*b1~a4*b4]
      第二次循环：
          va=[a5~a8]
          vb=[b5~b8]
          sum=[a1*b1+a5*b5~~~a4*b4+a8*b8]
      最后返回结果就是:
          res = (a1*b1+a5*b5)+......+(a4*b4+a8*b8)

    这样就实现了求和
    */
    // 先将sum初始化为[0.0f,0.0f,0.0f,0.0f]
    float32x4_t sum = vdupq_n_f32(0.0f);
    // 每次处理4个float
    for (size_t i = 0; i < dim; i += 4) 
    {
        // 读向量a的4个float
        float32x4_t va = vld1q_f32(a + i);
        // 同上
        float32x4_t vb = vld1q_f32(b + i);
        // 将va和vb的元素逐个相乘并累加放在sum内
        sum = vmlaq_f32(sum, va, vb);
    }
    // 将sum内的四个元素求和后返回
    return vaddvq_f32(sum);
}

inline int32_t simd_dot_product_u8_96(const uint8_t* a, const uint8_t* b) 
{
    // 依然初始化为[0,0,0,0]
    int32x4_t sum = vmovq_n_s32(0);
    // 每次处理16个uint8,一共6次即可完成
    for (int i = 0; i < 96; i += 16) 
    {
        // 加载16个uint8数据
        uint8x16_t va_u8 = vld1q_u8(a + i);
        // 同上
        uint8x16_t vb_u8 = vld1q_u8(b + i);
        // 转换为int8（需要减去128的偏移量，因为范围从0~255变为-128~127）
        int8x16_t va = vsubq_s8(vreinterpretq_s8_u8(va_u8), vdupq_n_s8(128));
        int8x16_t vb = vsubq_s8(vreinterpretq_s8_u8(vb_u8), vdupq_n_s8(128));
        // 拆分成a0~a7 这样的8个int8相乘
        int16x8_t prod_low = vmull_s8(vget_low_s8(va), vget_low_s8(vb));
        // 拆分成a8~a15 这样的8个int8相乘
        int16x8_t prod_high = vmull_s8(vget_high_s8(va), vget_high_s8(vb));
        // 相加并拓展为 int32类型数据
        // 从2*8个int16元素的变为1*4个int32元素
        sum = vaddq_s32(sum, vpaddlq_s16(prod_low));
        sum = vaddq_s32(sum, vpaddlq_s16(prod_high));
    }
    // 合并四个累加的结果
    int32_t sum_total = vgetq_lane_s32(sum, 0) + vgetq_lane_s32(sum, 1) 
                      + vgetq_lane_s32(sum, 2) + vgetq_lane_s32(sum, 3);
    return sum_total;
}

template<size_t Rerank = 20>
MinQueue<float, int> simd_hybrid_search(
    const uint8_t* base_q, const uint8_t* query_q,
    const float* base_orig, const float* query_orig,
    size_t base_number, size_t dim, size_t k
) 
{
    // stage1: 快速搜索（其实就是类似之前的最小堆保留topK结果）
    MinQueue<int32_t, int> first_stage;
    for (size_t i = 0; i < base_number; i++) 
    {
        int32_t ip = simd_dot_product_u8_96(base_q + i * dim, query_q);
        
        first_stage.emplace(ip, i);
        if (first_stage.size() > Rerank) 
        {
            first_stage.pop();
        }
    }

    // stage2:精准排序（借助float类型的内积来方便精准排序）
    MinQueue<float, int> final_result;
    while (!first_stage.empty()) 
    {
        // 从第一阶段的MinQueue取索引
        int index = first_stage.top().second;
        // 计算其对应的float点积
        float score = exact_dot_product(base_orig + index * dim,query_orig,dim);
        // 将float点积插入MinQueue final_result
        final_result.emplace(score, index);
        if (final_result.size() > k) 
        {
            final_result.pop();
        }
        // 处理完过渡到下一个
        first_stage.pop();
    }
    // 最后的final_result就是改良版本
    return final_result;
}
#endif