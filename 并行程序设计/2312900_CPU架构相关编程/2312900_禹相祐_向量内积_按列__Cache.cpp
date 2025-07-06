#include <iostream>
#include <vector>
#include <chrono>


using namespace std;
using namespace chrono;


// 逐列访问的计算方式
vector<int> column_major_dot(const vector<vector<int>>& matrix, const vector<int>& vec) {
    int n = matrix.size();
    vector<int> result(n, 0);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            result[j] += matrix[i][j] * vec[i];
        }
    }
    return result;
}


// cache优化的计算方式（按行访问）
vector<int> row_major_dot(const vector<vector<int>>& matrix, const vector<int>& vec) {
    int n = matrix.size();
    vector<int> result(n, 0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            result[j] += matrix[i][j] * vec[i];
        }
    }
    return result;
}


int main() {
    for (int n = 1000; n <= 10000; n += 1000) {
        cout << "Matrix size: " << n << "x" << n << endl;
        vector<vector<int>> matrix(n, vector<int>(n, 2));
        vector<int> vec(n, 2);

        double total_time_col = 0.0;
        double total_time_row = 0.0;

        cout << "Column-major times (ms): ";
        for (int t = 0; t < 10; ++t) {
            auto start = high_resolution_clock::now();
            vector<int> result1 = column_major_dot(matrix, vec);
            auto end = high_resolution_clock::now();
            double time_taken = duration<double, milli>(end - start).count();
            total_time_col += time_taken;
            cout << time_taken << "ms ";
        }
        cout << "\nColumn-major average time: " << total_time_col / 10.0 << " ms" << endl;

        cout << "Row-major times (ms): ";
        for (int t = 0; t < 10; ++t) {
            auto start = high_resolution_clock::now();
            vector<int> result2 = row_major_dot(matrix, vec);
            auto end = high_resolution_clock::now();
            double time_taken = duration<double, milli>(end - start).count();
            total_time_row += time_taken;
            cout << time_taken << "ms ";
        }
        cout << "\nRow-major average time: " << total_time_row / 10.0 << " ms" << endl;

        cout << "-----------------------------" << endl;
    }
    return 0;
}
