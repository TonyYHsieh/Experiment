#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>
#include <hip/device_functions.h>
#include <hip/math_functions.h>
#include <hip/hip_ext_ocp.h>
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
#include <type_traits>

#define TEST_VALUE_INIT 0

struct hipblaslt_fp6x32 {
  uint8_t val0 : 6;
  uint8_t val1 : 6;
  uint8_t val2 : 6;
  uint8_t val3 : 6;
  uint8_t val4 : 6;
  uint8_t val5 : 6;
  uint8_t val6 : 6;
  uint8_t val7 : 6;
  uint8_t val8 : 6;
  uint8_t val9 : 6;
  uint8_t val10 : 6;
  uint8_t val11 : 6;
  uint8_t val12 : 6;
  uint8_t val13 : 6;
  uint8_t val14 : 6;
  uint8_t val15 : 6;
  uint8_t val16 : 6;
  uint8_t val17 : 6;
  uint8_t val18 : 6;
  uint8_t val19 : 6;
  uint8_t val20 : 6;
  uint8_t val21 : 6;
  uint8_t val22 : 6;
  uint8_t val23 : 6;
  uint8_t val24 : 6;
  uint8_t val25 : 6;
  uint8_t val26 : 6;
  uint8_t val27 : 6;
  uint8_t val28 : 6;
  uint8_t val29 : 6;
  uint8_t val30 : 6;
  uint8_t val31 : 6;
} __attribute__((packed));

template <typename T>
float getFloat(const std::vector<T>& data, int M, int K, int P, int m, int k)
{
    return float(data[m * K + k]);
}

template <>
float getFloat(const std::vector<__amd_fp4x2_storage_t>& data, int K, int M, int P, int k, int m)
{
    int KP = K / P;
    int ki = k / P;
    int km = k % P;

    __amd_fp4x2_storage_t f4x2 = data[m * KP + ki];
    __amd_floatx2_storage_t f32x2 = __amd_cvt_fp4x2_to_floatx2_scale(f4x2, __AMD_OCP_E2M1, 0);

    return float(f32x2[km]);
}

template <>
float getFloat(const std::vector<hipblaslt_fp6x32>& data, int K, int M, int P, int k, int m)
{
    int KP = K / P;
    int ki = k / P;
    int km = k % P;

    union {
        hipblaslt_fp6x32 h;
        __amd_fp6x32_storage_t a;
    } val;

    val.h = data[m * KP + ki];
    __amd_floatx32_storage_t f32x32 = __amd_cvt_fp6x32_to_floatx32_scale(val.a, __AMD_OCP_E2M3, 0);

    return float(f32x32[km]);
}

template <typename T>
void dumpBuffer(const char* title, const std::vector<T>& data, int M, int N, int P)
{
    std::cout << "----- " << title << " start -----" << std::endl;
    for (int n=0; n<N; n++)
    {
        for (int m=0 ; m<M; m++)
        {
            std::cout << float(getFloat(data, M, N, P, m, n)) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "----- " << title << " end -------" << std::endl << std::endl;
}

template<typename T>
T initPackData(std::mt19937& seed)
{
#if TEST_VALUE_INIT
    return T(1.0f);
#else
    return T(float(std::uniform_int_distribution<int>(-3, 3)(seed)));
#endif
}

template<>
__amd_fp4x2_storage_t initPackData<__amd_fp4x2_storage_t>(std::mt19937& seed)
{
#if TEST_VALUE_INIT
    return 0x22;
#else
    __amd_floatx2_storage_t f2;
    f2.x = float(std::uniform_int_distribution<int>(-3, 3)(seed));
    f2.y = float(std::uniform_int_distribution<int>(-3, 3)(seed));
    return __amd_cvt_floatx2_to_fp4x2_scale(f2, __AMD_OCP_E2M1, 0);
#endif
}

template<>
hipblaslt_fp6x32 initPackData<hipblaslt_fp6x32>(std::mt19937& seed)
{
#if TEST_VALUE_INIT
    hipblaslt_fp6x32 val;
    val.val0 = 8;
    val.val1 = 8;
    val.val2 = 8;
    val.val3 = 8;
    val.val4 = 8;
    val.val5 = 8;
    val.val6 = 8;
    val.val7 = 8;
    val.val8 = 8;
    val.val9 = 8;
    val.val10 = 8;
    val.val11 = 8;
    val.val12 = 8;
    val.val13 = 8;
    val.val14 = 8;
    val.val15 = 8;
    val.val16 = 8;
    val.val17 = 8;
    val.val18 = 8;
    val.val19 = 8;
    val.val20 = 8;
    val.val21 = 8;
    val.val22 = 8;
    val.val23 = 8;
    val.val24 = 8;
    val.val25 = 8;
    val.val26 = 8;
    val.val27 = 8;
    val.val28 = 8;
    val.val29 = 8;
    val.val30 = 8;
    val.val31 = 8;
    return val;
#else
    
    union {
        hipblaslt_fp6x32 h;
        __amd_fp6x32_storage_t a;
    } val;

    __amd_floatx32_storage_t f32x32;

    for (int i=0; i<32; i++)
        f32x32[i] = float(std::uniform_int_distribution<int>(-3, 3)(seed));

    val.a = __amd_cvt_floatx32_to_fp6x32_scale(f32x32, __AMD_OCP_E2M3, 0);

    return val.h;
#endif
}

template<typename T>
void initData(std::vector<T>& data, int M, int N, int P)
{
    static std::mt19937 seed(69069);

    for (int n=0; n<N; n++)
    {
        int MP = M/P;
        for (int mp=0 ; mp<MP; mp++)
        {
            data[mp + n * MP] = initPackData<T>(seed);
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
#if TEST_VALUE_INIT
            data[m+n*M] = 127;
#else
            data[m+n*M] = uint8_t(std::uniform_int_distribution<unsigned>(-3+127, 3+127)(seed));
#endif
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
void CPUMatMul(std::vector<float>& cpuC, const std::vector<T>& cpuA, const std::vector<uint8_t>& scaleA, const std::vector<T>& cpuB, const std::vector<uint8_t>& scaleB, int M, int N, int K, int P, int B)
{
    for(int n=0; n<N; n++)
    {
        for(int m=0; m<M; m++)
        {
            cpuC[n*M+m] = 0.0f;

            int KB = K / B;
            for(int kb=0; kb<KB; kb++)
            {
                float sA = scaleToFloat(scaleA[m*KB+kb]);
                float sB = scaleToFloat(scaleB[n*KB+kb]);
                float accm = 0.0f;

                for (int b=0; b<B; b++) {
                    int k = kb * B + b;
                    float valA = getFloat(cpuA, K, M, P, k, m);
                    float valB = getFloat(cpuB, K, N, P, k, n);
                    accm = accm + valA * valB;
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
void GPUMatMul(std::vector<float>& cpuC, std::vector<T>& cpuA, std::vector<uint8_t> scaleA, std::vector<T>& cpuB, std::vector<uint8_t> scaleB, int M, int N, int K, int P, int B)
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
    err = hipMalloc(&gpuA, sizeof(T) * K * M / P);
    err = hipMalloc(&gpuB, sizeof(T) * K * N / P);
    err = hipMalloc(&gpuSA, sizeof(uint8_t) * KB * N);
    err = hipMalloc(&gpuSB, sizeof(uint8_t) * KB * N);

    err = hipMemset(gpuC, 0, sizeof(float) * M * N);
    err = hipMemcpyHtoD(gpuA, cpuA.data(), sizeof(T) * K * M / P);
    err = hipMemcpyHtoD(gpuB, cpuB.data(), sizeof(T) * K * N / P);
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
void Sample(const std::uint32_t M, const std::uint32_t N, const std::uint32_t K, const std::uint32_t P, const std::uint32_t B)
{
    std::vector<float> cpuR(M * N, 0.0f);
    std::vector<float> cpuC(M * N, 0.0f);
    std::vector<T> cpuA(K * M / P);
    std::vector<T> cpuB(K * N / P);
    std::vector<uint8_t> cpuAScale(K / B * M);
    std::vector<uint8_t> cpuBScale(K / B * N);

    initData(cpuA, K, M, P);
    initData(cpuB, K, N, P);

    initScale(cpuAScale, K / B, M);
    initScale(cpuBScale, K / B, N);

    CPUMatMul(cpuR, cpuA, cpuAScale, cpuB, cpuBScale, M, N, K, P, B);
    GPUMatMul(cpuC, cpuA, cpuAScale, cpuB, cpuBScale, M, N, K, P, B);

    dumpBuffer("A", cpuA, K, M, P);
    dumpBuffer("B", cpuB, K, N, P);
    dumpBuffer("C", cpuC, M, N, 1);
    dumpBuffer("R", cpuR, M, N, 1);

    validate(cpuR, cpuC, M, N);
}

int main(int argc, char **argv) {

    if (argc != 1)
    {
        std::cout << "./matmul" << std::endl;
        return -1;
    }

    Sample<hipblaslt_fp6x32>(16, 16, 128, 32, 32);

    return 0;
}
