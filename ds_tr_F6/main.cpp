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

struct uint6x16
{
    uint8_t val0  : 6;
    uint8_t val1  : 6;
    uint8_t val2  : 6;
    uint8_t val3  : 6;
    uint8_t val4  : 6;
    uint8_t val5  : 6;
    uint8_t val6  : 6;
    uint8_t val7  : 6;
    uint8_t val8  : 6;
    uint8_t val9  : 6;
    uint8_t val10 : 6;
    uint8_t val11 : 6;
    uint8_t val12 : 6;
    uint8_t val13 : 6;
    uint8_t val14 : 6;
    uint8_t val15 : 6;
} __attribute__((packed));

template <typename T>
void dumpBuffer(const char* title, const std::vector<T>& data, int M, int N, int P)
{
    std::cout << "----- " << title << " start -----" << std::endl;
    for (int n=0; n<N; n++)
    {
        int MP = M / P;
        for (int mp=0 ; mp<MP; mp++)
        {
            std::cout << float(data[mp+n*MP].val0)  << " ";
            std::cout << float(data[mp+n*MP].val1)  << " ";
            std::cout << float(data[mp+n*MP].val2)  << " ";
            std::cout << float(data[mp+n*MP].val3)  << " ";
            std::cout << float(data[mp+n*MP].val4)  << " ";
            std::cout << float(data[mp+n*MP].val5)  << " ";
            std::cout << float(data[mp+n*MP].val6)  << " ";
            std::cout << float(data[mp+n*MP].val7)  << " ";
            std::cout << float(data[mp+n*MP].val8)  << " ";
            std::cout << float(data[mp+n*MP].val9)  << " ";
            std::cout << float(data[mp+n*MP].val10) << " ";
            std::cout << float(data[mp+n*MP].val11) << " ";
            std::cout << float(data[mp+n*MP].val12) << " ";
            std::cout << float(data[mp+n*MP].val13) << " ";
            std::cout << float(data[mp+n*MP].val14) << " ";
            std::cout << float(data[mp+n*MP].val15) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "----- " << title << " end -------" << std::endl << std::endl;
}

template<typename T>
void initData(std::vector<T>& data, int M, int K, int P)
{
    uint32_t val = 0;

    for (int k=0; k<K; k++)
    {
        int MP = M / P;
        for (int mp=0 ; mp<MP; mp++)
        {
            for (int p=0; p<P; p++)
            {
                switch (p)
                {
                case 0:
                    data[mp+k*MP].val0 = (val % 64);
                    val++;
                break;
                case 1:
                    data[mp+k*MP].val1 = (val % 64);
                    val++;
                break;
                case 2:
                    data[mp+k*MP].val2 = (val % 64);
                    val++;
                break;
                case 3:
                    data[mp+k*MP].val3 = (val % 64);
                    val++;
                break;
                case 4:
                    data[mp+k*MP].val4 = (val % 64);
                    val++;
                break;
                case 5:
                    data[mp+k*MP].val5 = (val % 64);
                    val++;
                break;
                case 6:
                    data[mp+k*MP].val6 = (val % 64);
                    val++;
                break;
                case 7:
                    data[mp+k*MP].val7 = (val % 64);
                    val++;
                break;
                case 8:
                    data[mp+k*MP].val8 = (val % 64);
                    val++;
                break;
                case 9:
                    data[mp+k*MP].val9 = (val % 64);
                    val++;
                break;
                case 10:
                    data[mp+k*MP].val10 = (val % 64);
                    val++;
                break;
                case 11:
                    data[mp+k*MP].val11 = (val % 64);
                    val++;
                break;
                case 12:
                    data[mp+k*MP].val12 = (val % 64);
                    val++;
                break;
                case 13:
                    data[mp+k*MP].val13 = (val % 64);
                    val++;
                break;
                case 14:
                    data[mp+k*MP].val14 = (val % 64);
                    val++;
                break;
                case 15:
                    data[mp+k*MP].val15 = (val % 64);
                    val++;
                break;
                default:
                    data[mp+k*MP].val0  = 0;
                    data[mp+k*MP].val1  = 0;
                    data[mp+k*MP].val2  = 0;
                    data[mp+k*MP].val3  = 0;
                    data[mp+k*MP].val4  = 0;
                    data[mp+k*MP].val5  = 0;
                    data[mp+k*MP].val6  = 0;
                    data[mp+k*MP].val7  = 0;
                    data[mp+k*MP].val8  = 0;
                    data[mp+k*MP].val9  = 0;
                    data[mp+k*MP].val10 = 0;
                    data[mp+k*MP].val11 = 0;
                    data[mp+k*MP].val12 = 0;
                    data[mp+k*MP].val13 = 0;
                    data[mp+k*MP].val14 = 0;
                    data[mp+k*MP].val15 = 0;
                }
            }
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
void GPUTR(std::vector<T>& out, std::vector<T>& in, int M, int K, int P)
{
    hipDevice_t dev{};
    auto err = hipDeviceGet(&dev, 0);

    T* gpuOut = nullptr;
    T* gpuIn  = nullptr;

    err = hipMalloc(&gpuOut, sizeof(T) * K * M / P);
    err = hipMalloc(&gpuIn,  sizeof(T) * M * K / P);

    err = hipMemcpyHtoD(gpuIn, in.data(), sizeof(T) * M * K / P);

    hipModule_t module{};
    hipFunction_t func{};

    err = prepareASMKernel("Transpose", "build/transpose.co", &module, &func);
    if (err)
        std::cout << "find asm kernel failed" << std::endl;

    err = launchASMTR(func, gpuOut, gpuIn, M, K);
    if (err)
        std::cout << "launchASMTR error : " << err << std::endl;

    err = hipMemcpyDtoH(out.data(), gpuOut, sizeof(T) * K * M / P);

    err = hipModuleUnload(module);
    err = hipFree(gpuOut);
    err = hipFree(gpuIn);
}

void Sample()
{
    int M = 16;
    int K = 64;
    int P = 16;

    std::vector<uint6x16> in(sizeof(uint6x16) * M * K / P);
    std::vector<uint6x16> out(sizeof(uint6x16) * K * M / P);

    initData(in, M, K, P);

    GPUTR(out, in, M, K, P);

    dumpBuffer("In", in, M, K, P);
    dumpBuffer("Out", out, K, M, P);
}

int main(int argc, char **argv) {

    if (argc != 1)
    {
        std::cout << "./ds_tr_f8" << std::endl;
        return -1;
    }

    Sample();

    return 0;

}
