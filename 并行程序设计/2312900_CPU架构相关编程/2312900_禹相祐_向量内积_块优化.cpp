#include <vector>
#include <iostream>
#include <chrono>


using namespace std;


// 块处理优化的矩阵乘法
vector<int> block_optimized_calculate(const vector<vector<int>>& matrix, const vector<int>& vec, int block_size) {
    int n = matrix.size();
    vector<int> result(n, 0);


    // 块处理循环
    for (int block_i = 0; block_i < n; block_i += block_size) {
        for (int block_j = 0; block_j < n; block_j += block_size) {
            // 计算当前块的乘积
            for (int i = block_i; i < min(block_i + block_size, n); ++i) {
                for (int j = block_j; j < min(block_j + block_size, n); ++j) {
                    result[j] += matrix[i][j] * vec[i];
                }
            }
        }
    }


    return result;
}


int main() {
    // 块大小
    int block_size = 256;


    // 从 1000x1000 到 10000x10000 逐步测试
    for (int n = 1000; n <= 10000; n += 1000) {
        // 创建矩阵和向量
        vector<vector<int>> matrix(n, vector<int>(n, 1));  // 所有元素为 1 的矩阵
        vector<int> vec(n, 1);  // 所有元素为 1 的向量


        // 用来记录每次执行的时间
        double total_time = 0;


        // 运行 10 次并取平均
        for (int i = 0; i < 10; ++i) {
            // 开始计时
            auto start = chrono::high_resolution_clock::now();


            // 使用块处理优化的矩阵计算
            vector<int> result = block_optimized_calculate(matrix, vec, block_size);


            // 结束计时
            auto end = chrono::high_resolution_clock::now();
            chrono::duration<double> duration = end - start;


            // 累加每次的时间
            total_time += duration.count();
        }


        // 计算平均时间
        double average_time = (total_time / 10) * 1000;  // 转为毫秒


        // 输出每个矩阵大小的平均计算时间
        cout << "Matrix size: " << n << "x" << n << ", Average Time: " << average_time << " ms" << endl;
    }


    return 0;
}
