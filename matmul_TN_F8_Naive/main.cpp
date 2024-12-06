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

void initScale(std::vector<uint8_t>& data, int M, int N)
{
    static std::mt19937 seed(69069);

    for (int n=0; n<N; n++)
    {
        for (int m=0 ; m<M; m++)
        {
//            data[m+n*M] = 127;
            data[m+n*M] = uint8_t(std::uniform_int_distribution<unsigned int>(-3+127, 3+127)(seed));
        }
    }
}

float scaleToFloat(uint8_t u)
{
    union {
        uint32_t u;
        float f;
    } val;

    val.u = (u << 23);
    return val.f;
}

template <typename T>
void CPUMatMul(std::vector<float>& cpuC, const std::vector<T>& cpuA, const std::vector<uint8_t> scaleA, const std::vector<T>& cpuB, const std::vector<uint8_t> scaleB, int M, int N, int K, int B)
{
    for(int n=0; n<N; n++)
    {
        for(int m=0; m<M; m++)
        {
            int KB = K/B;
            for(int kb=0; kb<KB; kb++)
            {
                float sA = scaleToFloat(scaleA[m*KB+kb]);
                float sB = scaleToFloat(scaleB[n*KB+kb]);
                float accm = 0.0f;

                for (int b=0; b<B; b++) {
                    int k = kb*B+b;
                    accm = accm + (float(cpuA[m*K+k]) * float(cpuB[n*K+k]));
                }

                cpuC[n*M+m] += (sA * sB * accm);
            }
        }
    }
}

template<typename T>
hipError_t launchASMMatMul(hipFunction_t func, float* gpuC, T* gpuA, uint8_t* gpuSA, T* gpuB, uint8_t* gpuSB, int M, int N, int K)
{
    std::uint32_t workgroups = 1;

    KernelArguments args;
    args.append(gpuC);
    args.append(gpuA);
    args.append(gpuSA);
    args.append(gpuB);
    args.append(gpuSB);
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
void GPUMatMul(std::vector<float>& cpuC, std::vector<T>& cpuA, std::vector<uint8_t> scaleA, std::vector<T>& cpuB, std::vector<uint8_t> scaleB, int M, int N, int K, int B)
{
    int KB = K / B;

    hipDevice_t dev{};
    auto err = hipDeviceGet(&dev, 0);

    float*   gpuC  = nullptr;
    T*       gpuA  = nullptr;
    T*       gpuB  = nullptr;
    uint8_t* gpuSA = nullptr;
    uint8_t* gpuSB = nullptr;

    err = hipMalloc(&gpuC, sizeof(float) * M * N);
    err = hipMalloc(&gpuA, sizeof(T) * K * M);
    err = hipMalloc(&gpuB, sizeof(T) * K * N);
    err = hipMalloc(&gpuSA, sizeof(uint8_t) * KB * M);
    err = hipMalloc(&gpuSB, sizeof(uint8_t) * KB * N);

    err = hipMemset(gpuC, 0, sizeof(float) * M * N);
    err = hipMemcpyHtoD(gpuA, cpuA.data(), sizeof(T) * K * M);
    err = hipMemcpyHtoD(gpuB, cpuB.data(), sizeof(T) * K * N);
    err = hipMemcpyHtoD(gpuSA, scaleA.data(), sizeof(uint8_t) * KB * M);
    err = hipMemcpyHtoD(gpuSB, scaleB.data(), sizeof(uint8_t) * KB * N);

    hipModule_t module{};
    hipFunction_t func{};

    err = prepareASMKernel("MatMul", "build/matmul.co", &module, &func);
    if (err)
        std::cout << "find asm kernel failed" << std::endl;

    err = launchASMMatMul(func, gpuC, gpuA, gpuSA, gpuB, gpuSB, M, N, K);
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
void Sample(const std::uint32_t M, const std::uint32_t N, const std::uint32_t K, const std::uint32_t B)
{
    std::vector<float> cpuR(M * N, 0.0f);
    std::vector<float> cpuC(M * N, 0.0f);
    std::vector<T> cpuA(K * M);
    std::vector<T> cpuB(K * N);
    std::vector<uint8_t> cpuAScale(K / B * M);
    std::vector<uint8_t> cpuBScale(K / B * N);

    initData(cpuA, K, M);
    initData(cpuB, K, N);

    initScale(cpuAScale, K / B, M);
    initScale(cpuBScale, K / B, N);

    CPUMatMul(cpuR, cpuA, cpuAScale, cpuB, cpuBScale, M, N, K, B);
    GPUMatMul(cpuC, cpuA, cpuAScale, cpuB, cpuBScale, M, N, K, B);

//    dumpBuffer("A", cpuA, K, M);
//    dumpBuffer("B", cpuB, K, N);
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

    Sample<__hip_fp8_e4m3>(16, 16, 128, 32);

    return 0;

}
