#include <vector>
#include <cstdint>
#include <unordered_map>
#include <cmath>
#include <limits>
#include <random>
#include <algorithm>
#include <queue>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <mpi.h>

// 使用 IVF + PQ 结构进行向量索引和近似搜索

// 每个 coarse centroid 的倒排表，保存属于该中心的 PQ 编码数据和对应向量编号
struct InvertedList {
    std::vector<std::vector<uint8_t>> pq_codes;  // 每个向量的 PQ 编码
    std::vector<int> ids;  // 原始向量在 base 中的编号
};

// IVF + PQ 索引结构体
struct IVFPQIndex {
    std::vector<std::vector<float>> coarse_centroids;      // coarse quantizer 聚类中心 (nlist x dim)
    std::vector<std::vector<std::vector<float>>> pq_codebooks; // PQ 子空间的 codebook (M x Ks x sub_dim)
    std::unordered_map<int, InvertedList> inverted_lists;  // 倒排表：coarse centroid id -> 向量PQ码和id
    int M;      // 子空间个数
    int Ks;     // 每个子空间的聚类数 (一般为256)
    int dim;    // 原始向量维度
};

// 欧氏距离平方计算函数
inline float l2_distance_sq(const std::vector<float>& a, const std::vector<float>& b) {
    float dist = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}

// KMeans 聚类：用于训练 coarse centroid 和 PQ 子码本
inline std::vector<std::vector<float>> kmeans(const std::vector<std::vector<float>>& data, int k, int dim, int n_iter = 50) {
    int n = static_cast<int>(data.size());
    std::vector<std::vector<float>> centroids(k, std::vector<float>(dim));
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> uni(0, n - 1);

    // 随机初始化中心点
    for (int i = 0; i < k; ++i) {
        centroids[i] = data[uni(rng)];
    }

    std::vector<int> assignments(n);
    for (int iter = 0; iter < n_iter; ++iter) {
        // 1. 分配每个点到最近中心
        for (int i = 0; i < n; ++i) {
            float best_dist = std::numeric_limits<float>::max();
            int best_c = 0;
            for (int j = 0; j < k; ++j) {
                float dist = l2_distance_sq(data[i], centroids[j]);
                if (dist < best_dist) {
                    best_dist = dist;
                    best_c = j;
                }
            }
            assignments[i] = best_c;
        }
        // 2. 更新中心
        std::vector<std::vector<float>> new_centroids(k, std::vector<float>(dim, 0));
        std::vector<int> counts(k, 0);
        for (int i = 0; i < n; ++i) {
            int c = assignments[i];
            for (int d = 0; d < dim; ++d)
                new_centroids[c][d] += data[i][d];
            counts[c]++;
        }
        for (int j = 0; j < k; ++j) {
            if (counts[j] > 0) {
                for (int d = 0; d < dim; ++d)
                    new_centroids[j][d] /= counts[j];
            }
        }
        centroids = new_centroids;
    }
    return centroids;
}

// 训练 PQ 子码本，每个子空间做一次独立的 KMeans
inline std::vector<std::vector<std::vector<float>>> train_product_quantizer(const std::vector<std::vector<float>>& base, int M, int Ks, int dim) {
    int sub_dim = dim / M;
    std::vector<std::vector<std::vector<float>>> codebooks(M);
    for (int m = 0; m < M; ++m) {
        std::vector<std::vector<float>> sub_vectors;
        for (const auto& vec : base) {
            std::vector<float> sub(vec.begin() + m * sub_dim, vec.begin() + (m + 1) * sub_dim);
            sub_vectors.push_back(sub);
        }
        codebooks[m] = kmeans(sub_vectors, Ks, sub_dim);
    }
    return codebooks;
}

// 对 residual 向量进行 PQ 编码
inline std::vector<uint8_t> encode_pq(const std::vector<float>& vec, const std::vector<std::vector<std::vector<float>>>& pq_codebooks, int M, int dim) {
    int sub_dim = dim / M;
    std::vector<uint8_t> code(M);
    for (int m = 0; m < M; ++m) {
        std::vector<float> sub(vec.begin() + m * sub_dim, vec.begin() + (m + 1) * sub_dim);
        float best_dist = std::numeric_limits<float>::max();
        int best_k = 0;
        for (size_t k = 0; k < pq_codebooks[m].size(); ++k) {
            float dist = l2_distance_sq(sub, pq_codebooks[m][k]);
            if (dist < best_dist) {
                best_dist = dist;
                best_k = static_cast<int>(k);
            }
        }
        code[m] = static_cast<uint8_t>(best_k);
    }
    return code;
}

// 构建 IVF+PQ 索引：包含 coarse quantizer 和 residual PQ 编码
inline IVFPQIndex build_ivf_pq_index(const std::vector<std::vector<float>>& base, int nlist, int M, int Ks, int dim) {
    IVFPQIndex index;
    index.dim = dim;
    index.M = M;
    index.Ks = Ks;
    index.coarse_centroids = kmeans(base, nlist, dim);

    // ---------- 第一步：为每个向量找到最近 coarse centroid，并收集 residual ---------- //
    std::vector<int> coarse_assign(base.size());
    std::vector<std::vector<float>> residuals;
    residuals.reserve(base.size());

    for (size_t i = 0; i < base.size(); ++i) {
        const auto& vec = base[i];
        int best_cid = 0;
        float best_dist = std::numeric_limits<float>::max();
        for (int c = 0; c < nlist; ++c) {
            float dist = l2_distance_sq(vec, index.coarse_centroids[c]);
            if (dist < best_dist) {
                best_dist = dist;
                best_cid = c;
            }
        }
        coarse_assign[i] = best_cid;

        std::vector<float> residual(dim);
        for (int d = 0; d < dim; ++d) {
            residual[d] = vec[d] - index.coarse_centroids[best_cid][d];
        }
        residuals.emplace_back(std::move(residual));
    }

    // ---------- 第二步：在 residual 空间训练 PQ ---------- //
    index.pq_codebooks = train_product_quantizer(residuals, M, Ks, dim);

    // ---------- 第三步：编码 residual 并填充倒排表 ---------- //
    for (size_t i = 0; i < base.size(); ++i) {
        const auto& residual = residuals[i];
        int cid = coarse_assign[i];

        std::vector<uint8_t> pq_code = encode_pq(residual, index.pq_codebooks, M, dim);

        index.inverted_lists[cid].pq_codes.push_back(std::move(pq_code));
        index.inverted_lists[cid].ids.push_back(static_cast<int>(i));
    }

    return index;
}

// 最小堆比较器：使得 priority_queue 变成小顶堆
struct MinHeapComparator {
    bool operator()(const std::pair<float, int>& a, const std::pair<float, int>& b) const {
        return a.first > b.first; // 最小的距离优先
    }
};

// 方便后续mpi并行返回min_heap
struct MinHeapComparatorIntFloat {
    bool operator()(const std::pair<int, float>& a, const std::pair<int, float>& b) const {
        return a.second > b.second; // 最小的距离优先
    }
};

// 非MPI版本的搜索函数
inline std::priority_queue<std::pair<int, float>, std::vector<std::pair<int, float>>, MinHeapComparatorIntFloat>
ivf_pq_search(const IVFPQIndex& index, const std::vector<float>& query, int topk, int nprobe) {
    int dim = index.dim;
    int M = index.M;
    int sub_dim = dim / M;

    // 1. 找最近的nprobe个coarse center
    std::vector<std::pair<int, float>> coarse_dists;
    for (size_t i = 0; i < index.coarse_centroids.size(); ++i) {
        float dist = l2_distance_sq(query, index.coarse_centroids[i]);
        coarse_dists.emplace_back(static_cast<int>(i), dist);
    }
    std::partial_sort(coarse_dists.begin(), coarse_dists.begin() + std::min(nprobe, static_cast<int>(coarse_dists.size())), 
                     coarse_dists.end(),
                     [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
                         return a.second < b.second;
                     });

    // 2. 搜索候选集合
    std::vector<std::pair<float, int>> results;
    for (int i = 0; i < std::min(nprobe, static_cast<int>(coarse_dists.size())); ++i) {
        int cid = coarse_dists[i].first;
        auto it = index.inverted_lists.find(cid);
        if (it == index.inverted_lists.end()) continue;

        // 计算residual subquery
        const std::vector<float>& centroid = index.coarse_centroids[cid];
        std::vector<std::vector<float>> residual_subqueries(M, std::vector<float>(sub_dim));
        for (int m = 0; m < M; ++m) {
            for (int d = 0; d < sub_dim; ++d) {
                residual_subqueries[m][d] = query[m * sub_dim + d] - centroid[m * sub_dim + d];
            }
        }

        const InvertedList& invlist = it->second;
        for (size_t j = 0; j < invlist.pq_codes.size(); ++j) {
            float pq_dist = 0.0f;
            for (int m = 0; m < M; ++m) {
                uint8_t code = invlist.pq_codes[j][m];
                const std::vector<float>& codeword = index.pq_codebooks[m][code];
                pq_dist += l2_distance_sq(residual_subqueries[m], codeword);
            }
            float full_dist = pq_dist + coarse_dists[i].second;
            results.emplace_back(full_dist, invlist.ids[j]);
        }
    }

    // 3. 返回topk结果
    std::partial_sort(results.begin(), 
                     results.begin() + std::min(topk, static_cast<int>(results.size())), 
                     results.end(),
                     [](const std::pair<float, int>& a, const std::pair<float, int>& b) { 
                         return a.first < b.first; 
                     });

    std::priority_queue<std::pair<int, float>, std::vector<std::pair<int, float>>, MinHeapComparatorIntFloat> result_heap;
    for (int i = 0; i < topk && i < static_cast<int>(results.size()); ++i) {
        result_heap.emplace(results[i].second, results[i].first);
    }
    return result_heap;
}

// 计算查询向量与 PQ 编码向量的近似距离（考虑residual）
inline float compute_pq_distance(const IVFPQIndex& index, const std::vector<float>& query, 
                                const std::vector<uint8_t>& pq_code, int cid) {
    int M = index.M;
    int dim = index.dim;
    int sub_dim = dim / M;
    
    // 计算residual subquery
    const std::vector<float>& centroid = index.coarse_centroids[cid];
    float total_dist = 0.0f;
    
    for (int m = 0; m < M; ++m) {
        // 计算当前子空间的residual query
        std::vector<float> residual_subquery(sub_dim);
        for (int d = 0; d < sub_dim; ++d) {
            residual_subquery[d] = query[m * sub_dim + d] - centroid[m * sub_dim + d];
        }
        
        // 获取对应的codeword
        uint8_t code = pq_code[m];
        const std::vector<float>& codeword = index.pq_codebooks[m][code];
        
        // 计算residual subquery与codeword的距离
        total_dist += l2_distance_sq(residual_subquery, codeword);
    }
    
    return total_dist;
}

// 注意这里堆存储的是 pair<距离, 编号>，便于最终 merge
using MinHeap = std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, MinHeapComparator>;

// MPI_Search
inline std::priority_queue<std::pair<int, float>, std::vector<std::pair<int, float>>, MinHeapComparatorIntFloat>
ivf_pq_search_mpi(const IVFPQIndex& index, const std::vector<float>& query, int topk, int nprobe) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // 拿到当前进程编号
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // 拿到总共的进程数

    // 只在第一次调用时输出一些参数信息，方便调试
    static bool first_call = true;
    if (first_call && rank == 0) {
        std::cerr << "MPI Size: " << size << ", nprobe: " << nprobe << ", topk: " << topk << std::endl;
        first_call = false;
    }

    const int dim = index.dim;
    const int nlist = index.coarse_centroids.size();  // 倒排表的数量（即 coarse centroid 数量）

    // 所有进程各自计算 query 到所有 coarse centroid 的距离
    // 因为中心数量通常不大，这部分不并行也无妨
    std::vector<std::pair<float, int>> centroid_dists;
    for (int i = 0; i < nlist; ++i) {
        float dist = l2_distance_sq(query, index.coarse_centroids[i]);
        centroid_dists.emplace_back(dist, i);  // 存下距离和索引
    }

    // 只保留距离最近的 nprobe 个中心（就是 IVF 中的搜索范围）
    std::partial_sort(centroid_dists.begin(), centroid_dists.begin() + nprobe, centroid_dists.end());

    std::vector<int> selected_lists;
    for (int i = 0; i < nprobe; ++i) {
        selected_lists.push_back(centroid_dists[i].second);
    }

    // 每个进程处理一部分倒排表（均分 selected_lists）
    std::vector<std::pair<float, int>> local_results;
    local_results.reserve(nprobe * 1000);  // 预分配内存，省得反复 realloc

    for (size_t list_idx = rank; list_idx < selected_lists.size(); list_idx += size) {
        int lid = selected_lists[list_idx];
        auto it = index.inverted_lists.find(lid);
        if (it == index.inverted_lists.end()) continue;  // 该列表可能是空的，跳过

        const auto& invlist = it->second;
        const int list_size = static_cast<int>(invlist.pq_codes.size());
        if (list_size == 0) continue;

        // 开始构建 LUT ——用于快速查 PQ encoding 对应的距离
        const int M = index.M;
        const int sub_dim = dim / M;
        const int Ks = index.Ks;
        const std::vector<float>& centroid = index.coarse_centroids[lid];

        // 把原始 query 减去 coarse centroid 得到 residual
        std::vector<std::vector<float>> residual_subqueries(M, std::vector<float>(sub_dim));
        for (int m = 0; m < M; ++m) {
            for (int d = 0; d < sub_dim; ++d) {
                residual_subqueries[m][d] = query[m * sub_dim + d] - centroid[m * sub_dim + d];
            }
        }

        // LUT 构建：对每个子空间的 residual 向量，计算到所有 PQ codeword 的距离
        std::vector<std::vector<float>> dist_table(M, std::vector<float>(Ks));
        for (int m = 0; m < M; ++m) {
            for (int k = 0; k < Ks; ++k) {
                if (k < static_cast<int>(index.pq_codebooks[m].size())) {
                    dist_table[m][k] = l2_distance_sq(residual_subqueries[m], index.pq_codebooks[m][k]);
                } else {
                    dist_table[m][k] = 1e8f;  // 防止越界访问
                }
            }
        }

        // 使用 LUT 快速解码 + 累加距离
        for (int i = 0; i < list_size; ++i) {
            float pq_dist = 0.0f;
            const auto& code = invlist.pq_codes[i];
            for (int m = 0; m < M; ++m) {
                pq_dist += dist_table[m][code[m]];
            }
            local_results.emplace_back(pq_dist, invlist.ids[i]);  // 存入候选结果
        }
    }

    // 为了 recall，高 rank 的进程可能多保留一些候选结果（而不是只留 topk）
    const int keep_factor = std::max(3, 20 / size);  // 越多进程，每个保留越少
    const int local_keep = std::min(static_cast<int>(local_results.size()), topk * keep_factor);

    // 保留前 local_keep 个最小距离的结果
    if (local_results.size() > local_keep) {
        std::partial_sort(local_results.begin(),
                          local_results.begin() + local_keep,
                          local_results.end(),
                          [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                              return a.first < b.first;
                          });
        local_results.resize(local_keep);
    } else {
        std::sort(local_results.begin(), local_results.end(),
                 [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                     return a.first < b.first;
                 });
    }

    // 所有进程一次性互相广播结果（Allgather 比多轮 Gather 更高效）
    const int max_keep = topk * keep_factor;
    std::vector<float> all_dists(size * max_keep);
    std::vector<int> all_ids(size * max_keep);
    std::vector<float> padded_dists(max_keep, 1e8f);
    std::vector<int> padded_ids(max_keep, -1);

    // 先把自己的结果填入
    for (size_t i = 0; i < local_results.size() && i < max_keep; ++i) {
        padded_dists[i] = local_results[i].first;
        padded_ids[i] = local_results[i].second;
    }

    // MPI_Allgather：所有进程同步所有其他进程的结果（每个进程都能看到所有人数据）
    MPI_Allgather(padded_dists.data(), max_keep, MPI_FLOAT,
                  all_dists.data(), max_keep, MPI_FLOAT, MPI_COMM_WORLD);
    MPI_Allgather(padded_ids.data(), max_keep, MPI_INT,
                  all_ids.data(), max_keep, MPI_INT, MPI_COMM_WORLD);

    // 现在，每个进程手里都拿到了所有候选结果，可以各自排序，无需 broadcast
    std::vector<std::pair<float, int>> all_results;
    for (int p = 0; p < size; ++p) {
        for (int i = 0; i < max_keep; ++i) {
            int idx = p * max_keep + i;
            if (all_ids[idx] != -1) {
                all_results.emplace_back(all_dists[idx], all_ids[idx]);
            }
        }
    }

    // 选出 topk 全局最小的距离
    if (all_results.size() > topk) {
        std::partial_sort(all_results.begin(),
                         all_results.begin() + topk,
                         all_results.end(),
                         [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                             return a.first < b.first;
                         });
        all_results.resize(topk);
    } else {
        std::sort(all_results.begin(), all_results.end(),
                 [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                     return a.first < b.first;
                 });
    }

    // id & dis拆开，用来后面返回result
    std::vector<float> final_dists(topk, 0.0f);
    std::vector<int> final_ids(topk, -1);
    for (size_t i = 0; i < all_results.size() && i < topk; ++i) {
        final_dists[i] = all_results[i].first;
        final_ids[i] = all_results[i].second;
    }

    // 如果是第一次查询，打印几个结果看看（调试用）
    if (first_call && rank == 0) {
        std::cerr << "Final results:" << std::endl;
        for (int i = 0; i < std::min(3, topk); ++i) {
            if (final_ids[i] != -1) {
                std::cerr << "  ID: " << final_ids[i] << ", Dist: " << final_dists[i] << std::endl;
            }
        }
    }

    std::priority_queue<std::pair<int, float>, std::vector<std::pair<int, float>>, MinHeapComparatorIntFloat> result;
    for (int i = 0; i < topk; ++i) {
        if (final_ids[i] != -1) {
            int dist_int = static_cast<int>(final_dists[i] * 1000); // 距离保留 3 位小数
            result.emplace(dist_int, static_cast<float>(final_ids[i]));
        }
    }

    return result;
}

// 保存索引到文件
inline void save_index(const IVFPQIndex& index, const std::string& filename) {
    std::ofstream ofs(filename.c_str(), std::ios::binary);
    if (!ofs) {
        throw std::runtime_error("无法打开文件写入");
    }

    // 写入基础参数
    ofs.write(reinterpret_cast<const char*>(&index.dim), sizeof(index.dim));
    ofs.write(reinterpret_cast<const char*>(&index.M), sizeof(index.M));
    ofs.write(reinterpret_cast<const char*>(&index.Ks), sizeof(index.Ks));

    // 写入 coarse_centroids
    size_t nlist = index.coarse_centroids.size();
    ofs.write(reinterpret_cast<const char*>(&nlist), sizeof(nlist));
    for (const auto& centroid : index.coarse_centroids) {
        ofs.write(reinterpret_cast<const char*>(centroid.data()), centroid.size() * sizeof(float));
    }

    // 写入 pq_codebooks
    size_t M = index.pq_codebooks.size();
    ofs.write(reinterpret_cast<const char*>(&M), sizeof(M));
    for (const auto& subspace_codebook : index.pq_codebooks) {
        size_t Ks = subspace_codebook.size();
        ofs.write(reinterpret_cast<const char*>(&Ks), sizeof(Ks));
        for (const auto& codeword : subspace_codebook) {
            ofs.write(reinterpret_cast<const char*>(codeword.data()), codeword.size() * sizeof(float));
        }
    }

    // 写入倒排表数量
    size_t inverted_list_count = index.inverted_lists.size();
    ofs.write(reinterpret_cast<const char*>(&inverted_list_count), sizeof(inverted_list_count));

    for (const auto& entry : index.inverted_lists) {
        int cid = entry.first;
        const auto& invlist = entry.second;
        // 写入倒排表对应的coarse centroid id
        ofs.write(reinterpret_cast<const char*>(&cid), sizeof(cid));

        // 写入 pq_codes 数量
        size_t pq_codes_num = invlist.pq_codes.size();
        ofs.write(reinterpret_cast<const char*>(&pq_codes_num), sizeof(pq_codes_num));

        for (const auto& code : invlist.pq_codes) {
            ofs.write(reinterpret_cast<const char*>(code.data()), code.size() * sizeof(uint8_t));
        }

        // 写入 ids 数量（应与 pq_codes_num 一致）
        size_t ids_num = invlist.ids.size();
        ofs.write(reinterpret_cast<const char*>(&ids_num), sizeof(ids_num));
        ofs.write(reinterpret_cast<const char*>(invlist.ids.data()), ids_num * sizeof(int));
    }

    ofs.close();
}

// 从文件加载索引
inline IVFPQIndex load_index(const std::string& filename) {
    std::ifstream ifs(filename.c_str(), std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("无法打开索引文件");
    }

    IVFPQIndex index;

    // 读基础参数
    ifs.read(reinterpret_cast<char*>(&index.dim), sizeof(index.dim));
    ifs.read(reinterpret_cast<char*>(&index.M), sizeof(index.M));
    ifs.read(reinterpret_cast<char*>(&index.Ks), sizeof(index.Ks));

    // 读 coarse_centroids
    size_t nlist = 0;
    ifs.read(reinterpret_cast<char*>(&nlist), sizeof(nlist));
    index.coarse_centroids.resize(nlist, std::vector<float>(index.dim));
    for (auto& centroid : index.coarse_centroids) {
        ifs.read(reinterpret_cast<char*>(centroid.data()), centroid.size() * sizeof(float));
    }

    // 读 pq_codebooks
    size_t M = 0;
    ifs.read(reinterpret_cast<char*>(&M), sizeof(M));
    index.pq_codebooks.resize(M);
    int sub_dim = index.dim / index.M;
    for (auto& subspace_codebook : index.pq_codebooks) {
        size_t Ks = 0;
        ifs.read(reinterpret_cast<char*>(&Ks), sizeof(Ks));
        subspace_codebook.resize(Ks, std::vector<float>(sub_dim));
        for (auto& codeword : subspace_codebook) {
            ifs.read(reinterpret_cast<char*>(codeword.data()), codeword.size() * sizeof(float));
        }
    }

    // 读倒排表数量
    size_t inverted_list_count = 0;
    ifs.read(reinterpret_cast<char*>(&inverted_list_count), sizeof(inverted_list_count));

    for (size_t i = 0; i < inverted_list_count; ++i) {
        int cid = 0;
        ifs.read(reinterpret_cast<char*>(&cid), sizeof(cid));
        InvertedList invlist;

        size_t pq_codes_num = 0;
        ifs.read(reinterpret_cast<char*>(&pq_codes_num), sizeof(pq_codes_num));
        invlist.pq_codes.resize(pq_codes_num, std::vector<uint8_t>(index.M));
        for (auto& code : invlist.pq_codes) {
            ifs.read(reinterpret_cast<char*>(code.data()), code.size() * sizeof(uint8_t));
        }

        size_t ids_num = 0;
        ifs.read(reinterpret_cast<char*>(&ids_num), sizeof(ids_num));
        invlist.ids.resize(ids_num);
        ifs.read(reinterpret_cast<char*>(invlist.ids.data()), ids_num * sizeof(int));

        index.inverted_lists[cid] = std::move(invlist);
    }

    ifs.close();
    return index;
}




