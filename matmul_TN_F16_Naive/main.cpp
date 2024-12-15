#include <hip/hip_runtime.h>
#include <hip/device_functions.h>
#include <hip/hip_ext.h>
#include <hip/math_functions.h>
#include <hip/hip_fp8.h>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <iostream>
#include <limits>
#include <string>
#include <numeric>
#include <cstdint>
#include <cstring>
#include <random>
#include "KernelArguments.hpp"
#include "cblas.h"

template <typename T>
void dumpBuffer(const char* title, const std::vector<T>& data, int M, int N)
{
    std::cout << "----- " << title << " start -----" << std::endl;
    for (int n=0; n<N; n++)
    {
        for (int m=0 ; m<M; m++)
        {
            std::cout << float(data[m+n*M]) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "----- " << title << " end -------" << std::endl << std::endl;
}

template<typename T>
void initData(std::vector<T>& data, int M, int N)
{
    static std::mt19937 seed(69069);

    for (int n=0; n<N; n++)
    {
        for (int m=0 ; m<M; m++)
        {
//            data[m+n*M] = T(1.0f);
            data[m+n*M] = T(float(std::uniform_int_distribution<int>(-3, 3)(seed)));
        }
    }
}

template <typename T>
void CPUMatMul(std::vector<float>& cpuC, const std::vector<T>& cpuA, const std::vector<T>& cpuB,  int M, int N, int K)
{
    for(int n=0; n<N; n++)
    {
        for(int m=0; m<M; m++)
        {
            cpuC[n*M+m] = 0;
            for(int k=0; k<K; k++)
            {
                cpuC[n*M+m] += (float(cpuA[m*K+k]) * float(cpuB[n*K+k]));
            }
        }
    }
}

template<typename T>
hipError_t launchASMMatMul(hipFunction_t func, float* gpuC, T* gpuA, T* gpuB, int M, int N, int K)
{
    std::uint32_t workgroups = 1;

    KernelArguments args;
    args.append(gpuC);
    args.append(gpuA);
    args.append(gpuB);
    args.append(M);
    args.append(N);
    args.append(K);
    args.applyAlignment();

    std::size_t argsSize = args.size();
    void *launchArgs[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER,
        args.buffer(),
        HIP_LAUNCH_PARAM_BUFFER_SIZE,
        &argsSize,
        HIP_LAUNCH_PARAM_END
    };

    hipStream_t stream{};
    auto err = hipStreamCreate(&stream);

    err = hipExtModuleLaunchKernel(func, 64 * workgroups, 1, 1, 64, 1, 1, 1000 * sizeof(float), nullptr, nullptr, launchArgs);

    err = hipStreamSynchronize(stream);

    err = hipStreamDestroy(stream);

    return err;
}


hipError_t prepareASMKernel(const std::string &funcName, const std::string &coPath, hipModule_t *module, hipFunction_t *func) {
    auto err = hipModuleLoad(module, coPath.c_str());
    if (err != hipSuccess)
        std::cout << "hipModuleLoad failed" << std::endl;
    err = hipModuleGetFunction(func, *module, funcName.c_str());
    if (err != hipSuccess)
        std::cout << "hipModuleGetFunction failed" << std::endl;
    return err;
}

template <typename T>
void GPUMatMul(std::vector<float>& cpuC, std::vector<T>& cpuA, std::vector<T>& cpuB, int M, int N, int K)
{
    hipDevice_t dev{};
    auto err = hipDeviceGet(&dev, 0);

    float*   gpuC  = nullptr;
    T*       gpuA  = nullptr;
    T*       gpuB  = nullptr;

    err = hipMalloc(&gpuC, sizeof(float) * M * N);
    err = hipMalloc(&gpuA, sizeof(T) * K * M);
    err = hipMalloc(&gpuB, sizeof(T) * K * N);

    err = hipMemset(gpuC, 0, sizeof(float) * M * N);
    err = hipMemcpyHtoD(gpuA, cpuA.data(), sizeof(T) * K * M);
    err = hipMemcpyHtoD(gpuB, cpuB.data(), sizeof(T) * K * N);

    hipModule_t module{};
    hipFunction_t func{};

    err = prepareASMKernel("MatMul", "build/matmul.co", &module, &func);
    if (err)
        std::cout << "find asm kernel failed" << std::endl;

    err = launchASMMatMul(func, gpuC, gpuA, gpuB, M, N, K);
    if (err)
        std::cout << "launchASMMatMul error : " << err << std::endl;

    err = hipMemcpyDtoH(cpuC.data(), gpuC, sizeof(float) * M * N);

    err = hipModuleUnload(module);
    err = hipFree(gpuC);
    err = hipFree(gpuA);
    err = hipFree(gpuB);
}

template <typename T>
void validate(const std::vector<T>& cpuR, const std::vector<T>& cpuC, int M, int N)
{
    for (int n=0; n<N; n++)
    {
        for (int m=0 ; m<M; m++)
        {
            float err = std::abs(float(cpuR[m+n*M]) - float(cpuC[m+n*M]));
            if (err > 1e-5)
            {
                std::cout << "Error " << err << " Ref " << float(cpuR[m+n*M]) << " GPU " << cpuC[m+n*M] << std::endl;
                return;
            }
        }
    }
    std::cout << "PASS" << std::endl;
}

template <typename T>
void Sample(const std::uint32_t M, const std::uint32_t N, const std::uint32_t K)
{
    std::vector<float> cpuR(M * N, 0.0f);
    std::vector<float> cpuC(M * N, 0.0f);
    std::vector<T> cpuA(K * M);
    std::vector<T> cpuB(K * N);

    initData(cpuA, K, M);
    initData(cpuB, K, N);

    CPUMatMul(cpuR, cpuA, cpuB, M, N, K);
    GPUMatMul(cpuC, cpuA, cpuB, M, N, K);

    dumpBuffer("A", cpuA, K, M);
    dumpBuffer("B", cpuB, K, N);
    dumpBuffer("C", cpuC, M, N);
    dumpBuffer("R", cpuR, M, N);
 
    validate(cpuR, cpuC, M, N);
}

int main(int argc, char **argv) {

    if (argc != 1)
    {
        std::cout << "./matmul" << std::endl;
        return -1;
    }

    Sample<half>(16, 16, 32);

    return 0;

}
