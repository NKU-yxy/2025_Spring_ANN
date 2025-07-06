// Total Main.cc
/*
写于Pthread和OpenMP实验：

ps:若要测试，
    eg.1.要测试openmp并行化加速的话，需要将openmp的头文件包含进来并且注释掉Pthread和普通IVF+PQ的头文件（因为存在相同名字的索引构建函数会报错）
       并且将运行的时候的代码，即auto res = ...... 选择openmp的运行代码并且注释掉其他俩;
       2.可以调整对应参数并且再生成一个索引并存在files下，也可以调用已经生成了的索引（若想要生成新的索引需要将line284、287解除注释并对应修改287和291的对应索引的名字TT）
    

*/
#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <sys/time.h>
#include <omp.h>
#include "hnswlib/hnswlib/hnswlib.h"
#include "flat_scan.h"

#include <queue>
#include <cstdint>

// 可以自行添加需要的头文件

// ------------SIMD实验部分------------------------//
// 优化1
#include "flat_search_sq.h"

// 优化2
#include "simd.h"
#include <arm_neon.h>  // NEON SIMD 头文件，用于 ARM 平台加速

// ---------------------------------------------//


// ----------Pthread和OpenMP实验部分------------//
// 普通的 IVF+PQ
//#include "ivfpq_search.h"

// Pthread编程
//#include "pthread_search.h"
//#include <pthread.h>

// OpenMP编程
#include "openmp_search.h"
#include <omp.h>

//-----------------------------------------//


using namespace hnswlib;

template<typename T>
T *LoadData(std::string data_path, size_t& n, size_t& d)
{
    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
    fin.read((char*)&n,4);
    fin.read((char*)&d,4);
    T* data = new T[n*d];
    int sz = sizeof(T);
    for(int i = 0; i < n; ++i){
        fin.read(((char*)data + i*d*sz), d*sz);
    }
    fin.close();

    std::cerr<<"load data "<<data_path<<"\n";
    std::cerr<<"dimension: "<<d<<"  number:"<<n<<"  size_per_element:"<<sizeof(T)<<"\n";

    return data;
}

struct SearchResult
{
    float recall;
    int64_t latency; // 单位us
};

void build_index(float* base, size_t base_number, size_t vecdim)
{
    const int efConstruction = 150; // 为防止索引构建时间过长，efc建议设置200以下
    const int M = 16; // M建议设置为16以下

    HierarchicalNSW<float> *appr_alg;
    InnerProductSpace ipspace(vecdim);
    appr_alg = new HierarchicalNSW<float>(&ipspace, base_number, M, efConstruction);

    appr_alg->addPoint(base, 0);
    #pragma omp parallel for
    for(int i = 1; i < base_number; ++i) {
        appr_alg->addPoint(base + 1ll*vecdim*i, i);
    }

    char path_index[1024] = "files/hnsw.index";
    appr_alg->saveIndex(path_index);
}


int main(int argc, char *argv[])
{
   
    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;

    std::string data_path = "/anndata/"; 
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);
    // 只测试前2000条查询
    test_number = 2000;

    const size_t k = 10;

    std::vector<SearchResult> results;
    results.resize(test_number);

// -----------------------------------------------------------
// 优化1的改动代码    
    /*
    // 优化1 部分
    // max_abs 获取改为
    float max_abs_base = find_max_abs(base, base_number, vecdim);
    float max_abs_query = find_max_abs(test_query, test_number, vecdim);
    float max_abs = std::max(max_abs_base, max_abs_query);


    std::cerr << "Max absolute value from base: " << max_abs << std::endl;
    std::vector<uint8_t> base_q(base_number * vecdim);
    std::vector<uint8_t> query_q(test_number * vecdim);
    change_dataset(base, base_q.data(), base_number, vecdim, max_abs);
    change_dataset(test_query, query_q.data(), test_number, vecdim, max_abs);
    */

// -----------------------------------------------------------
// 优化2的改动代码    
/*
    // 因为知道了数据分布 就直接写上max abs了
    const float GLOBAL_MAX_ABS = 0.53f;
    std::vector<uint8_t> base_q(base_number * vecdim);
    std::vector<uint8_t> query_q(test_number * vecdim);
    std::vector<float> mean_base, mean_query;

    simd_quantize_global(base, base_q.data(), base_number, vecdim, 
                        GLOBAL_MAX_ABS, mean_base);
    simd_quantize_global(test_query, query_q.data(), test_number, vecdim,
                        GLOBAL_MAX_ABS, mean_query);
*/



//----------------------读取索引结束-------------------------------------------------------//

    // 如果你需要保存索引，可以在这里添加你需要的函数，你可以将下面的注释删除来查看pbs是否将build.index返回到你的files目录中
    // 要保存的目录必须是files/*
    // 每个人的目录空间有限，不需要的索引请及时删除，避免占空间太大
    // 不建议在正式测试查询时同时构建索引，否则性能波动会较大
    // 下面是一个构建hnsw索引的示例
    // build_index(base, base_number, vecdim);

/*
// ---------------------- ivf pq here ---------------------------------//
// ----------------------M=32 ----------------------------------//
// nprobe: 对应idx00
    30  0.995  10333 
    20  0.987  5935
    10  0.961  3096

// ----------------------M=48 ----------------------------------//
// nprobe: 对应idx01
    30  0.995  19594
    20  0.988  13836
    10  0.961  6834

// -----------------------M=96 ----------------------------------//
// nprobe: 对应idx02
    30  0.995  35754
    20  0.987  26510
    10  0.961  13592
*/

// --------------------- my code here --------------------------------//

    // ------------M=32的时候-----------------------------------------
    // 128:  0.809  1103.77us    对应idx09
    // 256： 0.81 684.5us 对应idx07
    // 512: 0.808 779us 对应idx08
    // --------------------------------------------------------

    // -----------M=32-----------------------------------//
    // nlist =128,nprobe=30  0.812 2235
    // nlist =128,nprobe=25  0.811 1595
    // nlist =128,nprobe=20  0.809 1370
    // nlist=128,nprobe=15   0.806 1362
    // nlist=128,nprobe=13  
    // nlist=128,nprobe=12   
    // nlist=128,nprobe=10  0.799  1003
    // nlist=128,nprobe=8  
    // nlist=128,nprobe=7  
    // nlist=128,nprobe=5   0.767 683


    // 到时候画个图就好
    // ----------- M=48 ---------------------------------//
    // nlist =128,nprobe=30   0.905 4452    对应idx10
    // nlist =128,nprobe=25  0.903  3636
    // nlist =128,nprobe=20  0.90  2900 
    // nlist=128,nprobe=15  0.896  2700
    // nlist=128,nprobe=13  0.893  2250
    // nlist=128,nprobe=12  0.892  2150   star this !!!!!! ！！！！！！！！！！
    // nlist=128,nprobe=10  0.885  2054
    // nlist=128,nprobe=8  0.874   2014
    // nlist=128,nprobe=7  0.866  1400
    // nlist=128,nprobe=5   0.84   1336

    // ---------- M=96试试--------------------//
    // 对应idx11
    // nlist=128,nprobe=30 0.976 8312
    // nlist=128,nprobe=25 0.974 7266
    // nlist=128,nprobe=20 0.97  5600
    // nlist=128,nprobe=15 0.963 4800
    // nlist=128,nprobe=13 0.958 3980
    // nlist=128,nprobe=12 0.955 3800
    // nlist=128,nprobe=10 0.947 3600
    // nlist=128,nprobe=8  0.933 3150  
    // nlist=128,nprobe=7  0.922 2331   
    // nlist=128,nprobe=5  0.889  2200
    int nlist = 128;

    int M = 96;
    int Ks = 256;
    int dim = 96;
    // 原先参数为20
    int nprobe = 10;

/*
    // ---------------- OPENMP部分 ---------------------- //
    // M=96  对应idx 12
    nprobe:
    30  0.981 6823
    25  0.979 5699
    20  0.975 5327
    15 0.967  3684
    10 0.951 3342
    5  0.892 1869

    // M=48 对应idx 13
    nprobe:
    30  0.903  3687
    25  0.901  3095
    20  0.898  2545
    15  0.893  1820
    10  0.882  1690
    05  0.836  1001

    // M=32 对应idx 14
    nprobe:
    30  0.792 1582
    25  0.792 1310
    20  0.789 1137
    15  0.786 799
    10  0.778 751
    05  0.746 450

*/
    // ivfpq的数据处理 ------------------------------------------------
    std::vector<std::vector<float>> base_vec(base_number, std::vector<float>(dim));
    for (int i = 0; i < base_number; ++i)  
        for (int j = 0; j < dim; ++j)
            base_vec[i][j] = base[i * dim + j];
    
    // -----------------------------------------------------------------
    
    // ------------------- 创建新的索引并且保存-------------------------
    // 新建索引
    //IVFPQIndex iiidx = build_ivf_pq_index(base_vec, nlist, M, Ks, dim);

    // 保存索引
    //save_index(iiidx,"files/idx12");

    // ------------------若需要调用以前的那就直接加载就好了~--------------------

    // 加载索引
    IVFPQIndex my_index = load_index("files/idx12");
 
    // 查询测试代码
    for(int i = 0; i < test_number; ++i) {
        const unsigned long Converter = 1000 * 1000;
        struct timeval val;
        int ret = gettimeofday(&val, NULL);

        // 该文件已有代码中你只能修改该函数的调用方式
        // 可以任意修改函数名，函数参数或者改为调用成员函数，但是不能修改函数返回值。


        /* SIMD部分
        // 默认的调用方式
       // auto res = flat_search(base, test_query + i*vecdim, base_number, vecdim, k);

        // 优化1的调用方式
        // 优化1: sq: 函数改为flat_search_sq
        //auto res = flat_search_sq(base_q.data(), query_q.data() + i * vecdim,base_number, vecdim, k, max_abs);
                
        // 优化2的调用方式
        //auto res = simd_hybrid_search(base_q.data(),query_q.data() + i * vecdim,base,test_query + i * vecdim,base_number, vecdim, k);
        
        */
        
        // ivf+pq转类型
        std::vector<float> query_vec(dim);
        for (int j = 0; j < dim; ++j)
            query_vec[j] = test_query[i * dim + j];

        // openmp优化
        // 目前参数：M=96 nprobe=10 nlist=128 Ks=256
        auto res = openmp_ivf_pq_search(my_index,query_vec,k,nprobe);

        // pthread优化
        //auto res = pthread_ivf_pq_search(my_index,query_vec,k,nprobe);

        // 普通IVF+PQ
        //auto res = ivf_pq_search(my_index,query_vec,k,nprobe);

        struct timeval newVal;
        ret = gettimeofday(&newVal, NULL);
        int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);

        std::set<uint32_t> gtset;
        for(int j = 0; j < k; ++j){
            int t = test_gt[j + i*test_gt_d];
            gtset.insert(t);
        }

        size_t acc = 0;

        while (res.size()) {   
            int x = res.top().second;
            if(gtset.find(x) != gtset.end()){
                ++acc;
            }
            res.pop();
        }
        float recall = (float)acc/k;

        results[i] = {recall, diff};
    }

    float avg_recall = 0, avg_latency = 0;
    for(int i = 0; i < test_number; ++i) {
        avg_recall += results[i].recall;
        avg_latency += results[i].latency;
    }

    // 浮点误差可能导致一些精确算法平均recall不是1
    std::cout << "average recall: "<<avg_recall / test_number<<"\n";
    std::cout << "average latency (us): "<<avg_latency / test_number<<"\n";
    return 0;

}
