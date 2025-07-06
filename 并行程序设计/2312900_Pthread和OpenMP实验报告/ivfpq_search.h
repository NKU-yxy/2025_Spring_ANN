#ifndef IVF_PQ_SEARCH_H
#define IVF_PQ_SEARCH_H

#include <vector>
#include <cstdint>
#include <unordered_map>
#include <cmath>
#include <limits>
#include <random>
#include <algorithm>

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
float l2_distance_sq(const std::vector<float>& a, const std::vector<float>& b) {
    float dist = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}

// KMeans 聚类：用于训练 coarse centroid 和 PQ 子码本
std::vector<std::vector<float>> kmeans(const std::vector<std::vector<float>>& data, int k, int dim, int n_iter = 20) {
    int n = data.size();
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
std::vector<std::vector<std::vector<float>>> train_product_quantizer(const std::vector<std::vector<float>>& base, int M, int Ks, int dim) {
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
std::vector<uint8_t> encode_pq(const std::vector<float>& vec, const std::vector<std::vector<std::vector<float>>>& pq_codebooks, int M, int dim) {
    int sub_dim = dim / M;
    std::vector<uint8_t> code(M);
    for (int m = 0; m < M; ++m) {
        std::vector<float> sub(vec.begin() + m * sub_dim, vec.begin() + (m + 1) * sub_dim);
        float best_dist = std::numeric_limits<float>::max();
        int best_k = 0;
        for (int k = 0; k < pq_codebooks[m].size(); ++k) {
            float dist = l2_distance_sq(sub, pq_codebooks[m][k]);
            if (dist < best_dist) {
                best_dist = dist;
                best_k = k;
            }
        }
        code[m] = static_cast<uint8_t>(best_k);
    }
    return code;
}

// 构建 IVF+PQ 索引：包含 coarse quantizer 和 residual PQ 编码
IVFPQIndex build_ivf_pq_index(const std::vector<std::vector<float>>& base, int nlist, int M, int Ks, int dim) {
    IVFPQIndex index;
    index.dim = dim;
    index.M = M;
    index.Ks = Ks;
    index.coarse_centroids = kmeans(base, nlist, dim);
    index.pq_codebooks = train_product_quantizer(base, M, Ks, dim);

    for (int i = 0; i < base.size(); ++i) {
        const auto& vec = base[i];

        // 1. 找到与当前向量最接近的 coarse centroid
        int best_cid = 0;
        float best_dist = std::numeric_limits<float>::max();
        for (int c = 0; c < nlist; ++c) {
            float dist = l2_distance_sq(vec, index.coarse_centroids[c]);
            if (dist < best_dist) {
                best_dist = dist;
                best_cid = c;
            }
        }

        // 2. 计算 residual（原始向量减去 coarse centroid）
        std::vector<float> residual(dim);
        for (int d = 0; d < dim; ++d) {
            residual[d] = vec[d] - index.coarse_centroids[best_cid][d];
        }

        // 3. 对 residual 向量编码为 PQ code
        std::vector<uint8_t> pq_code = encode_pq(residual, index.pq_codebooks, M, dim);

        // 4. 存入对应 coarse centroid 的倒排表
        index.inverted_lists[best_cid].pq_codes.push_back(pq_code);
        index.inverted_lists[best_cid].ids.push_back(i);
    }
    return index;
}

// 最小堆比较器：使得 priority_queue 变成小顶堆
struct MinHeapComparator {
    bool operator()(const std::pair<int, float>& a, const std::pair<int, float>& b) const {
        return a.first > b.first; // 最小的距离优先
    }
};

// IVF + PQ 搜索：返回支持 push/pop/top 的真正最小堆
std::priority_queue<std::pair<int, float>, std::vector<std::pair<int, float>>, MinHeapComparator>
ivf_pq_search(const IVFPQIndex& index, const std::vector<float>& query, int topk, int nprobe) {
    int dim = index.dim;
    int M = index.M;
    int Ks = index.Ks;
    int sub_dim = dim / M;

    // 1. 找最近的 nprobe 个 coarse center
    std::vector<std::pair<int, float>> coarse_dists;
    for (int i = 0; i < index.coarse_centroids.size(); ++i) {
        float dist = l2_distance_sq(query, index.coarse_centroids[i]);
        coarse_dists.emplace_back(i, dist);
    }
    std::partial_sort(coarse_dists.begin(), coarse_dists.begin() + nprobe, coarse_dists.end(),
        [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
            return a.second < b.second;
        });

    // 2. 计算 residual subqueries
    std::vector<std::vector<float>> residual_subqueries(M, std::vector<float>(sub_dim));

    // 3. 进行 PQ 近似计算并加入堆
    std::priority_queue<std::pair<int, float>, std::vector<std::pair<int, float>>, MinHeapComparator> result_heap;

    for (int i = 0; i < nprobe; ++i) {
        int cid = coarse_dists[i].first;
        float coarse_dist = coarse_dists[i].second;  // 新增
        const std::vector<float>& centroid = index.coarse_centroids[cid];
        const InvertedList& invlist = index.inverted_lists.at(cid);

        // 预处理 residual 向量
        for (int m = 0; m < M; ++m) {
            for (int d = 0; d < sub_dim; ++d) {
                residual_subqueries[m][d] = query[m * sub_dim + d] - centroid[m * sub_dim + d];
            }
        }

        for (int j = 0; j < invlist.pq_codes.size(); ++j) {
            float pq_dist = 0.0f;
            for (int m = 0; m < M; ++m) {
                uint8_t code = invlist.pq_codes[j][m];
                const std::vector<float>& codeword = index.pq_codebooks[m][code];
                pq_dist += l2_distance_sq(residual_subqueries[m], codeword);
            }
           // float approx_dist = pq_dist;  // 关键修改
            result_heap.emplace(static_cast<int>(pq_dist),invlist.ids[j]);
        }
    }

    return result_heap;
}

//----------------------index------------------------------------------//

// 保存索引到文件
void save_index(const IVFPQIndex& index, const std::string& filename) {
    std::ofstream ofs(filename, std::ios::binary);
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
        int cid =entry.first;
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
IVFPQIndex load_index(const std::string& filename) {
    std::ifstream ifs(filename, std::ios::binary);
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

#endif // IVF_PQ_SEARCH_H