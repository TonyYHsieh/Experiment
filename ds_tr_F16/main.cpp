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
void initData(std::vector<T>& data, int M, int K)
{
    for (int k=0; k<K; k++)
    {
        for (int m=0 ; m<M; m++)
        {
            data[m+k*M] = T(m+k*M);
        }
    }
}

template<typename T>
hipError_t launchASMTR(hipFunction_t func, T* gpuOut, T* gpuIn, int M, int K)
{
    std::uint32_t workgroups = 1;

    KernelArguments args;
    args.append(gpuOut);
    args.append(gpuIn);
    args.append(M);
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
void GPUTR(std::vector<T>& out, std::vector<T>& in, int M, int K)
{
    hipDevice_t dev{};
    auto err = hipDeviceGet(&dev, 0);

    T* gpuOut = nullptr;
    T* gpuIn  = nullptr;

    err = hipMalloc(&gpuOut, sizeof(T) * K * M);
    err = hipMalloc(&gpuIn,  sizeof(T) * M * K);

    err = hipMemcpyHtoD(gpuIn, in.data(), sizeof(T) * M * K);

    hipModule_t module{};
    hipFunction_t func{};

    err = prepareASMKernel("Transpose", "build/transpose.co", &module, &func);
    if (err)
        std::cout << "find asm kernel failed" << std::endl;

    err = launchASMTR(func, gpuOut, gpuIn, M, K);
    if (err)
        std::cout << "launchASMTR error : " << err << std::endl;

    err = hipMemcpyDtoH(out.data(), gpuOut, sizeof(T) * K * M);

    err = hipModuleUnload(module);
    err = hipFree(gpuOut);
    err = hipFree(gpuIn);
}

void Sample()
{
    int M = 16;
    int K = 16;

    std::vector<uint16_t> in(M * K);
    std::vector<uint16_t> out(K * M);

    initData(in, M, K);

    GPUTR(out, in, M, K);

    dumpBuffer("In", in, M, K);
    dumpBuffer("Out", out, K, M);
}

int main(int argc, char **argv) {

    if (argc != 1)
    {
        std::cout << "./transpose" << std::endl;
        return -1;
    }

    Sample();

    return 0;

}
