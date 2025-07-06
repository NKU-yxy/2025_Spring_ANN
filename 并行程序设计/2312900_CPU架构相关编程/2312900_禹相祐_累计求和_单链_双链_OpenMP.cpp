#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>


using namespace std;
using namespace std::chrono;


// 最简单的单链路求和
long long normal_get_sum(const vector<int>& v)
{
    long long result = 0;
    for (int i = 0; i < v.size(); i++)
    {
        result += v[i];
    }
    return result;
}


// 二路链式相加求和
long long two_ways_get_sum(const vector<int>& v)
{
    long long odd_sum = 0, even_sum = 0;
    for (int i = 0; i < v.size(); i++)
    {
        // 偶数Sum
        if (i % 2 == 0)
            even_sum += v[i];
        // 奇数Sum
        else
            odd_sum += v[i];
    }
    return odd_sum + even_sum;
}



// OpenMP 方法
long long parallel_get_sum(const vector<int>& v)
{
    long long result = 0;
    #pragma omp parallel for reduction(+:result)
    for (int i = 0; i < v.size(); i++)
    {
        result += v[i];
    }
    return result;
}



// 拿来执行测试的函数
void operate_testing(int array_size)
{

    vector<int> my_array(array_size);
    for (int i = 0; i < array_size; i++)
    {
        my_array[i] = i;
    }

    long long avg_time_ordinary = 0, avg_time_two_way = 0, avg_time_parallel = 0;

    // 执行十次，求执行时间的平均值
    for (int i = 0; i < 10; i++)
    {
        // 单链方法
        auto start = high_resolution_clock::now();
        long long sum1 = normal_get_sum(my_array);
        auto end = high_resolution_clock::now();
        avg_time_ordinary += duration_cast<microseconds>(end - start).count();

        // 双链方法
        auto two_way_start = high_resolution_clock::now();
        long long sum2 = two_ways_get_sum(my_array);
        auto two_way_end = high_resolution_clock::now();
        avg_time_two_way += duration_cast<microseconds>(two_way_end - two_way_start).count();

        // OpenMP方法
        auto openmp_start = high_resolution_clock::now();
        long long sum3 = normal_get_sum(my_array);
        auto openmp_end = high_resolution_clock::now();
        avg_time_parallel += duration_cast<microseconds>(openmp_end - openmp_start).count();

    }

    cout << "数组大小: " << array_size << endl;
    cout << "单链平均时间: " << avg_time_ordinary / 10 << " ms" << endl;
    cout << "双链平均时间: " << avg_time_two_way / 10 << " ms" << endl;
    cout << "OpenMP求和平均时间: " << avg_time_parallel / 10 << " ms" << endl;
    cout << "------------------------------------------------" << endl;
}


int main()
{
    // 不同规模的array_size 从1K到1KW
    const vector<int> array_sizes = {1000, 10000, 100000, 1000000, 10000000};

    // 调用函数，执行测试
    for (int size : array_sizes)
    {
        operate_testing(size);
    }
    return 0;
}
