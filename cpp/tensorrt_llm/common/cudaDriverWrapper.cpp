/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define CUDA_LIB_NAME "cuda"

#if defined(_WIN32)
#include <windows.h>
#define dllOpen(name) LoadLibrary("nv" name ".dll")
#define dllClose(handle) FreeLibrary(static_cast<HMODULE>(handle))
#define dllGetSym(handle, name) static_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle), name))
#else // For non-Windows platforms
#include <dlfcn.h>
#define dllOpen(name) dlopen("lib" name ".so.1", RTLD_LAZY)
#define dllClose(handle) dlclose(handle)
#define dllGetSym(handle, name) dlsym(handle, name)
#endif // defined(_WIN32)

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaDriverWrapper.h"
#include "tensorrt_llm/common/logger.h"
#include <cuda.h>

#include <cstdio>
#include <mutex>

namespace tensorrt_llm::common
{

std::shared_ptr<CUDADriverWrapper> CUDADriverWrapper::getInstance()
{
    static std::mutex mutex;
    static std::weak_ptr<CUDADriverWrapper> instance;
    std::shared_ptr<CUDADriverWrapper> result = instance.lock();
    if (result)
    {
        return result;
    }

    std::lock_guard<std::mutex> const lock(mutex);
    result = instance.lock();
    if (!result)
    {
        result = std::shared_ptr<CUDADriverWrapper>(new CUDADriverWrapper());
        instance = result;
    }
    return result;
}

CUDADriverWrapper::CUDADriverWrapper()
    : handle(dllOpen(CUDA_LIB_NAME))
{

    TLLM_CHECK_WITH_INFO(handle != nullptr, "CUDA driver library is not open correctly.");

    auto load_sym = [](void* handle, char const* name)
    {
        void* ret = dllGetSym(handle, name);
        return ret;
    };

    *reinterpret_cast<void**>(&_cuGetErrorName) = load_sym(handle, "cuGetErrorName");
    *reinterpret_cast<void**>(&_cuGetErrorString) = load_sym(handle, "cuGetErrorString");
    *reinterpret_cast<void**>(&_cuFuncSetAttribute) = load_sym(handle, "cuFuncSetAttribute");
    *reinterpret_cast<void**>(&_cuLinkComplete) = load_sym(handle, "cuLinkComplete");
    *reinterpret_cast<void**>(&_cuModuleUnload) = load_sym(handle, "cuModuleUnload");
    *reinterpret_cast<void**>(&_cuLinkDestroy) = load_sym(handle, "cuLinkDestroy");
    *reinterpret_cast<void**>(&_cuModuleLoadData) = load_sym(handle, "cuModuleLoadData");
    *reinterpret_cast<void**>(&_cuLinkCreate) = load_sym(handle, "cuLinkCreate_v2");
    *reinterpret_cast<void**>(&_cuModuleGetFunction) = load_sym(handle, "cuModuleGetFunction");
    *reinterpret_cast<void**>(&_cuModuleGetGlobal) = load_sym(handle, "cuModuleGetGlobal_v2");
    *reinterpret_cast<void**>(&_cuLinkAddFile) = load_sym(handle, "cuLinkAddFile_v2");
    *reinterpret_cast<void**>(&_cuLinkAddData) = load_sym(handle, "cuLinkAddData_v2");
    *reinterpret_cast<void**>(&_cuLaunchCooperativeKernel) = load_sym(handle, "cuLaunchCooperativeKernel");
    *reinterpret_cast<void**>(&_cuLaunchKernel) = load_sym(handle, "cuLaunchKernel");
    *reinterpret_cast<void**>(&_cuLaunchKernelEx) = load_sym(handle, "cuLaunchKernelEx");
    *reinterpret_cast<void**>(&_cuTensorMapEncodeTiled) = load_sym(handle, "cuTensorMapEncodeTiled");
    *reinterpret_cast<void**>(&_cuMemcpyDtoH) = load_sym(handle, "cuMemcpyDtoH_v2");
    *reinterpret_cast<void**>(&_cuDeviceGetAttribute) = load_sym(handle, "cuDeviceGetAttribute");
    *reinterpret_cast<void**>(&_cuOccupancyMaxActiveClusters) = load_sym(handle, "cuOccupancyMaxActiveClusters");
    *reinterpret_cast<void**>(&_cuFuncGetParamInfo) = load_sym(handle, "cuFuncGetParamInfo");
    *reinterpret_cast<void**>(&_cuFuncGetAttribute) = load_sym(handle, "cuFuncGetAttribute");
}

CUDADriverWrapper::~CUDADriverWrapper()
{
    dllClose(handle);
}

CUresult CUDADriverWrapper::cuGetErrorName(CUresult error, char const** pStr) const
{
    return (*_cuGetErrorName)(error, pStr);
}

CUresult CUDADriverWrapper::cuGetErrorString(CUresult error, char const** pStr) const
{
    return (*_cuGetErrorString)(error, pStr);
}

CUresult CUDADriverWrapper::cuFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib, int value) const
{
    return (*_cuFuncSetAttribute)(hfunc, attrib, value);
}

CUresult CUDADriverWrapper::cuLinkComplete(CUlinkState state, void** cubinOut, size_t* sizeOut) const
{
    return (*_cuLinkComplete)(state, cubinOut, sizeOut);
}

CUresult CUDADriverWrapper::cuModuleUnload(CUmodule hmod) const
{
    return (*_cuModuleUnload)(hmod);
}

CUresult CUDADriverWrapper::cuLinkDestroy(CUlinkState state) const
{
    return (*_cuLinkDestroy)(state);
}

CUresult CUDADriverWrapper::cuModuleLoadData(CUmodule* module, void const* image) const
{
    return (*_cuModuleLoadData)(module, image);
}

CUresult CUDADriverWrapper::cuLinkCreate(
    unsigned int numOptions, CUjit_option* options, void** optionValues, CUlinkState* stateOut) const
{
    return (*_cuLinkCreate)(numOptions, options, optionValues, stateOut);
}

CUresult CUDADriverWrapper::cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, char const* name) const
{
    return (*_cuModuleGetFunction)(hfunc, hmod, name);
}

CUresult CUDADriverWrapper::cuModuleGetGlobal(CUdeviceptr* dptr, size_t* bytes, CUmodule hmod, char const* name) const
{
    return (*_cuModuleGetGlobal)(dptr, bytes, hmod, name);
}

CUresult CUDADriverWrapper::cuLinkAddFile(CUlinkState state, CUjitInputType type, char const* path,
    unsigned int numOptions, CUjit_option* options, void** optionValues) const
{
    return (*_cuLinkAddFile)(state, type, path, numOptions, options, optionValues);
}

CUresult CUDADriverWrapper::cuLinkAddData(CUlinkState state, CUjitInputType type, void* data, size_t size,
    char const* name, unsigned int numOptions, CUjit_option* options, void** optionValues) const
{
    return (*_cuLinkAddData)(state, type, data, size, name, numOptions, options, optionValues);
}

CUresult CUDADriverWrapper::cuLaunchCooperativeKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
    unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
    unsigned int sharedMemBytes, CUstream hStream, void** kernelParams) const
{
    return (*_cuLaunchCooperativeKernel)(
        f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams);
}

CUresult CUDADriverWrapper::cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
    unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
    unsigned int sharedMemBytes, CUstream hStream, void** kernelParams, void** extra) const
{
    return (*_cuLaunchKernel)(
        f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
}

namespace
{
std::string stringify_launch_config(CUlaunchConfig const& config, CUfunction f, void** kernelParams, void** extra, 
    CUDADriverWrapper const& wrapper)
{
    std::stringstream ss;

    // Grid dimensions
    ss << "Launch Configuration:\n";
    ss << "  Grid Dimensions: (" << config.gridDimX << ", " << config.gridDimY << ", " << config.gridDimZ << ")\n";

    // Block dimensions
    ss << "  Block Dimensions: (" << config.blockDimX << ", " << config.blockDimY << ", " << config.blockDimZ << ")\n";

    // Calculate total threads per block
    unsigned int threadsPerBlock = config.blockDimX * config.blockDimY * config.blockDimZ;
    
    // Get max threads per block
    int maxThreadsPerBlock = 0;
    wrapper.cuFuncGetAttribute(&maxThreadsPerBlock, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, f);
    ss << "  Threads Per Block: " << threadsPerBlock << " (Max allowed: " << maxThreadsPerBlock << ")\n";

    // Get number of registers
    int numRegs = 0;
    wrapper.cuFuncGetAttribute(&numRegs, CU_FUNC_ATTRIBUTE_NUM_REGS, f);
    ss << "  Registers Per Thread: " << numRegs << "\n";

    // Shared memory
    int maxSharedSize = 0;
    wrapper.cuFuncGetAttribute(&maxSharedSize, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, f);
    ss << "  Shared Memory: " << config.sharedMemBytes << " bytes (Max allowed: " << maxSharedSize << " bytes)\n";

    // Stream info
    ss << "  Stream: 0x" << std::hex << reinterpret_cast<uintptr_t>(config.hStream) << std::dec << "\n";

    // PDL (Param vs Extra) usage
    ss << "  Parameter Mode: " << (extra != nullptr ? "Extra (PDL)" : "KernelParams") << "\n";

    // Parameter count validation using cuFuncGetParamInfo
    size_t paramOffset = 0, paramSize = 0;
    CUresult paramInfoResult = wrapper.cuFuncGetParamInfo(f, 0, &paramOffset, &paramSize);
    if (paramInfoResult == CUDA_SUCCESS)
    {
        // Try to get info for second parameter - if it succeeds, there's more than one parameter
        size_t param2Offset = 0, param2Size = 0;
        if (wrapper.cuFuncGetParamInfo(f, 1, &param2Offset, &param2Size) == CUDA_SUCCESS)
        {
            ss << "  WARNING: Function has multiple parameters (expected single parameter)\n";
        }
        else
        {
            ss << "  Parameter validation: OK (single parameter)\n";
        }
    }
    else
    {
        ss << "  Parameter validation: Failed to get parameter info\n";
    }

    // Cluster dimensions
    bool hasClusterDim = false;
    for (uint i = 0; i < config.numAttrs; ++i)
    {
        CUlaunchAttribute const& attr = config.attrs[i];
        if (attr.id == CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION)
        {
            hasClusterDim = true;
            ss << "  Cluster Dimensions: (" << attr.value.clusterDim.x << ", " 
               << attr.value.clusterDim.y << ", " << attr.value.clusterDim.z << ")\n";
            break;
        }
    }
    if (!hasClusterDim)
    {
        ss << "  Cluster Dimensions: Not specified\n";
    }

    // Other attributes
    if (config.numAttrs > 0)
    {
        ss << "  Additional Attributes:\n";
        for (uint i = 0; i < config.numAttrs; ++i)
        {
            CUlaunchAttribute const& attr = config.attrs[i];
            if (attr.id != CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION)  // Skip cluster dim as it's already printed
            {
                ss << "    [" << i << "] ";
                switch (attr.id)
                {
                case CU_LAUNCH_ATTRIBUTE_PRIORITY: 
                    ss << "Priority: " << attr.value.priority;
                    break;
                default: 
                    ss << "Unknown Attribute (ID=" << attr.id << ")";
                    break;
                }
                ss << "\n";
            }
        }
    }

    return ss.str();
}
} // namespace

CUresult CUDADriverWrapper::cuLaunchKernelEx(
    CUlaunchConfig const* config, CUfunction f, void** kernelParams, void** extra) const
{
    // Validate configuration
    int maxThreadsPerBlock = 0;
    CUresult result = cuFuncGetAttribute(&maxThreadsPerBlock, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, f);
    if (result != CUDA_SUCCESS) return result;

    unsigned int threadsPerBlock = config->blockDimX * config->blockDimY * config->blockDimZ;
    if (threadsPerBlock > static_cast<unsigned int>(maxThreadsPerBlock))
    {
        TLLM_LOG_ERROR("Threads per block (%u) exceeds maximum allowed (%d)", 
            threadsPerBlock, maxThreadsPerBlock);
        return CUDA_ERROR_INVALID_VALUE;
    }

    int maxSharedSize = 0;
    result = cuFuncGetAttribute(&maxSharedSize, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, f);
    if (result != CUDA_SUCCESS) return result;

    if (config->sharedMemBytes > static_cast<unsigned int>(maxSharedSize))
    {
        TLLM_LOG_ERROR("Shared memory size (%u) exceeds maximum allowed (%d)", 
            config->sharedMemBytes, maxSharedSize);
        return CUDA_ERROR_INVALID_VALUE;
    }

    // Print detailed launch configuration
    TLLM_LOG_DEBUG("%s", stringify_launch_config(*config, f, kernelParams, extra, *this).c_str());

    TLLM_CHECK_DEBUG_WITH_INFO(
        (extra != nullptr) != (kernelParams != nullptr), 
        "Exactly one of 'extra' and 'kernelParams' should be set.");

    return (*_cuLaunchKernelEx)(config, f, kernelParams, extra);
}

CUresult CUDADriverWrapper::cuTensorMapEncodeTiled(CUtensorMap* tensorMap, CUtensorMapDataType tensorDataType,
    cuuint32_t tensorRank, void* globalAddress, cuuint64_t const* globalDim, cuuint64_t const* globalStrides,
    cuuint32_t const* boxDim, cuuint32_t const* elementStrides, CUtensorMapInterleave interleave,
    CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) const
{
    return (*_cuTensorMapEncodeTiled)(tensorMap, tensorDataType, tensorRank, globalAddress, globalDim, globalStrides,
        boxDim, elementStrides, interleave, swizzle, l2Promotion, oobFill);
}

CUresult CUDADriverWrapper::cuMemcpyDtoH(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount) const
{
    return (*_cuMemcpyDtoH)(dstHost, srcDevice, ByteCount);
}

CUresult CUDADriverWrapper::cuDeviceGetAttribute(int* pi, CUdevice_attribute attrib, CUdevice dev) const
{
    return (*_cuDeviceGetAttribute)(pi, attrib, dev);
}

CUresult CUDADriverWrapper::cuOccupancyMaxActiveClusters(
    int* maxActiveClusters, CUfunction f, CUlaunchConfig const* config) const
{
    return (*_cuOccupancyMaxActiveClusters)(maxActiveClusters, f, config);
}

CUresult CUDADriverWrapper::cuFuncGetParamInfo(
    CUfunction func, size_t paramIndex, size_t* paramOffset, size_t* paramSize) const
{
    return (*_cuFuncGetParamInfo)(func, paramIndex, paramOffset, paramSize);
}

CUresult CUDADriverWrapper::cuFuncGetAttribute(int* pi, CUfunction_attribute attrib, CUfunction hfunc) const
{
    return (*_cuFuncGetAttribute)(pi, attrib, hfunc);
}

} // namespace tensorrt_llm::common
