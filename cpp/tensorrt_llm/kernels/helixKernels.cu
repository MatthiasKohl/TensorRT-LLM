/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/kernels/helixKernels.h"

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

using namespace tensorrt_llm::common;

namespace tensorrt_llm
{
namespace kernels
{
// utility: async memcpy with LDGSTS
template <int Size = 16>
__device__ inline void memcpy_async_ldgsts(void* shared_ptr, void const* global_ptr)
{
    if constexpr (Size == 16)
    {
        asm volatile("cp.async.cg.shared.global [%0], [%1], %2, %2;"
                     :
                     /* no output */
                     : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(shared_ptr))),
                     "l"(static_cast<uint64_t>(__cvta_generic_to_global(global_ptr))), "n"(16)
                     : "memory");
    }
    else
    {
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %2;"
                     :
                     /* no output */
                     : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(shared_ptr))),
                     "l"(static_cast<uint64_t>(__cvta_generic_to_global(global_ptr))), "n"(Size)
                     : "memory");
    }
}

static constexpr int WARP_SIZE = 32;

// Utility: warp-level corrected sum
template <int N>
__device__ inline void warpReduceCorrectedSum(
    float (&correctedVal)[N], float (&maxVal)[N], float (&sumVal)[N], float scale)
{
    float warp_max = maxVal[0];
#pragma unroll
    for (int nn = 1; nn < N; ++nn)
        warp_max = fmaxf(warp_max, maxVal[nn]);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000 && defined(__CUDA_ARCH_FEAT_SM100_ALL))
    asm("redux.sync.max.f32 %0, %1, 0xffffffff;\n" : "=f"(warp_max) : "f"(warp_max));
#else
#pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset *= 2)
        warp_max = fmaxf(warp_max, __shfl_xor_sync(0xffffffff, warp_max, offset));
#endif
    float global_sum = 0.F;
    float corrected_max_exp[N];
#pragma unroll
    for (int nn = 0; nn < N; ++nn)
    {
        corrected_max_exp[nn] = sumVal[nn] * expf((maxVal[nn] - warp_max) * scale);
        global_sum += corrected_max_exp[nn];
    }
#pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset *= 2)
        global_sum += __shfl_xor_sync(0xffffffff, global_sum, offset);
    auto norm = 1.F / global_sum;
#pragma unroll
    for (int nn = 0; nn < N; ++nn)
        correctedVal[nn] = corrected_max_exp[nn] * norm;
}

// here we define the fallback kernel for the post-processing,
// if the main kernel uses too much shared memory

static constexpr int FB_MAX_CP_VAL_PER_THREAD = 8;
static constexpr int FB_MAX_CP = WARP_SIZE * FB_MAX_CP_VAL_PER_THREAD;
static constexpr int FB_BYTES_O_PER_THREAD = 16;
static constexpr int FB_NUM_PRE_LOAD = 8;

// Kernel: fused helix post-processing
// output: [num_tokens, num_heads * kv_lora_rank] (half)
// gathered_o: [cp_size, num_tokens, num_heads * kv_lora_rank] (half)
// gathered_stats: [cp_size, num_tokens, num_heads, 2] (fp32)
// note: we remove restrict from gathered_o to avoid compiler hoisting the barrier
// above loads of gathered_o
template <typename T>
__global__ void helix_postprocess_kernel_fallback(T* __restrict__ output, T const* /*__restrict__*/ gathered_o,
    float2 const* __restrict__ gathered_stats, int cp_size, int kv_lora_rank, float scale)
{
    // Each block processes one (token, head)
    // gridDim.x: num_tokens, gridDim.y: num_heads
    // there are two separate types of warps:
    // warp 0 calculates the correction values (one per cp_size)
    // all other warps pre-load the gathered_o elements for the current token/head
    // and once warp 0 is done, all other warps can start accumulating the output
    static constexpr int NUM_O_PER_THREAD = FB_BYTES_O_PER_THREAD / sizeof(T);
    static constexpr int NUM_PRE_LOAD = FB_NUM_PRE_LOAD;

    int tok_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int num_tokens = gridDim.x;
    int num_heads = gridDim.y;

    int const cp_size_aligned = ((cp_size + NUM_PRE_LOAD - 1) / NUM_PRE_LOAD) * NUM_PRE_LOAD;
    __shared__ float smem_correction[FB_MAX_CP];

    int lane_idx = threadIdx.x % WARP_SIZE;
    int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / WARP_SIZE, 0);
    if (warp_idx == 0)
    {
        // the warp collectively calculates the correction values
        float max_values[FB_MAX_CP_VAL_PER_THREAD];
        float sum_values[FB_MAX_CP_VAL_PER_THREAD];
#pragma unroll
        for (int cp_val_idx = 0; cp_val_idx < FB_MAX_CP_VAL_PER_THREAD; ++cp_val_idx)
        {
            auto cp_idx = cp_val_idx * WARP_SIZE + lane_idx;
            auto stats_offset = cp_idx * num_tokens * num_heads + tok_idx * num_heads + head_idx;
            float2 stats = cp_idx < cp_size ? gathered_stats[stats_offset] : make_float2(-INFINITY, 0.F);
            max_values[cp_val_idx] = stats.x;
            sum_values[cp_val_idx] = stats.y;
        }
        float corrected_values[FB_MAX_CP_VAL_PER_THREAD];
        warpReduceCorrectedSum(corrected_values, max_values, sum_values, scale);
#pragma unroll
        for (int cp_val_idx = 0; cp_val_idx < FB_MAX_CP_VAL_PER_THREAD; ++cp_val_idx)
        {
            auto cp_idx = cp_val_idx * WARP_SIZE + lane_idx;
            smem_correction[cp_idx] = corrected_values[cp_val_idx];
        }
        cg::this_thread_block().sync();
    }
    else
    {
        // all other warps pre-load the gathered_o elements for the current token/head
        auto const* gathered_o_off = gathered_o + tok_idx * num_heads * kv_lora_rank + head_idx * kv_lora_rank;
        // we subtract WARP_SIZE because first warp is not participating here
        gathered_o_off += (threadIdx.x - WARP_SIZE) * NUM_O_PER_THREAD;
        float4 const* gathered_o_16b = reinterpret_cast<float4 const*>(gathered_o_off);
        auto gathered_16b_stride = (num_tokens * num_heads * kv_lora_rank) / NUM_O_PER_THREAD;
        T vals[NUM_PRE_LOAD][NUM_O_PER_THREAD];
#pragma unroll
        for (int cp_idx = 0; cp_idx < NUM_PRE_LOAD && cp_idx < cp_size; ++cp_idx)
        {
            auto val
                = cp_idx < cp_size ? gathered_o_16b[cp_idx * gathered_16b_stride] : make_float4(0.F, 0.F, 0.F, 0.F);
            *reinterpret_cast<float4*>(vals[cp_idx]) = val;
        }
        float final_sum[NUM_O_PER_THREAD];
#pragma unroll
        for (int o_idx = 0; o_idx < NUM_O_PER_THREAD; ++o_idx)
        {
            final_sum[o_idx] = 0.F;
        }
        cg::this_thread_block().sync();

        // here we can trigger the dependent kernels to start
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
        cudaTriggerProgrammaticLaunchCompletion();
#endif

        float corr_vals[NUM_PRE_LOAD];
#pragma unroll
        for (int cp_idx = 0; cp_idx < NUM_PRE_LOAD && cp_idx < cp_size; ++cp_idx)
        {
            corr_vals[cp_idx] = smem_correction[cp_idx];
        }

        for (int cp_idx_base = NUM_PRE_LOAD; cp_idx_base < cp_size_aligned; cp_idx_base += NUM_PRE_LOAD)
        {
#pragma unroll
            for (int cp_idx = 0; cp_idx < NUM_PRE_LOAD; ++cp_idx)
            {
#pragma unroll
                for (int o_idx = 0; o_idx < NUM_O_PER_THREAD; ++o_idx)
                {
                    final_sum[o_idx] += static_cast<float>(vals[cp_idx][o_idx]) * corr_vals[cp_idx];
                }
            }
#pragma unroll
            for (int cp_idx = 0; cp_idx < NUM_PRE_LOAD; ++cp_idx)
            {
                *reinterpret_cast<float4*>(vals[cp_idx]) = cp_idx_base + cp_idx < cp_size
                    ? gathered_o_16b[(cp_idx_base + cp_idx) * gathered_16b_stride]
                    : make_float4(0.F, 0.F, 0.F, 0.F);
                corr_vals[cp_idx] = cp_idx_base + cp_idx < cp_size ? smem_correction[cp_idx_base + cp_idx] : 0.F;
            }
        }
#pragma unroll
        for (int cp_idx = 0; cp_idx < NUM_PRE_LOAD && cp_idx < cp_size; ++cp_idx)
        {
#pragma unroll
            for (int o_idx = 0; o_idx < NUM_O_PER_THREAD; ++o_idx)
            {
                final_sum[o_idx] += static_cast<float>(vals[cp_idx][o_idx]) * corr_vals[cp_idx];
            }
        }
        T output_typed[NUM_O_PER_THREAD];
#pragma unroll
        for (int o_idx = 0; o_idx < NUM_O_PER_THREAD; ++o_idx)
        {
            output_typed[o_idx] = static_cast<T>(final_sum[o_idx]);
        }
        auto* output_off = output + tok_idx * num_heads * kv_lora_rank + head_idx * kv_lora_rank;
        output_off += (threadIdx.x - WARP_SIZE) * NUM_O_PER_THREAD;
        *reinterpret_cast<float4*>(output_off) = *reinterpret_cast<float4*>(output_typed);
    }
}

// the main kernel
static constexpr int MIN_NUM_THREADS = 64;
static constexpr int MAX_NUM_THREADS = 256;
static constexpr int MAX_VAL_PER_THREAD = 4;

// Kernel: fused helix post-processing with async memcpy
// output: [num_tokens, num_heads * kv_lora_rank] (half)
// gathered_o: [cp_size, num_tokens, num_heads * kv_lora_rank] (half)
// gathered_stats: [cp_size, num_tokens, num_heads, 2] (fp32)
template <typename T>
__global__ void helix_postprocess_kernel(T* __restrict__ output, T const* __restrict__ gathered_o,
    float2 const* __restrict__ gathered_stats, int cp_size, int num_tokens, int num_heads, int num_heads_per_block,
    int kv_lora_rank, float scale)
{
    // Each block processes one token, and potentially multiple heads
    // gridDim.x: num_tokens
    // each warp processes one head at a time
    // Each thread processes elements over kv_lora_rank with loop unrolling

    int token_idx = blockIdx.x;
    int num_warps = blockDim.x / WARP_SIZE;
    int head_idx_base = blockIdx.y * num_warps;
    int head_idx_end = head_idx_base + num_heads_per_block;
    int lane_idx = threadIdx.x % WARP_SIZE;
    int warp_idx = threadIdx.x / WARP_SIZE;

    // size: num_warps * cp_size * kv_lora_rank
    // note: this must be 16-byte aligned (guaranteed by C++ standard)
    extern __shared__ char shared_mem[];
    T* smem_o = reinterpret_cast<T*>(shared_mem);
    // note: because kv_lora_rank * sizeof(T) is a multiple of 16,
    // we know that smem_stats is 16-byte aligned
    // size: num_warps * cp_size
    float2* smem_stats = reinterpret_cast<float2*>(smem_o + num_warps * cp_size * kv_lora_rank);

    for (int head_idx = head_idx_base + warp_idx; head_idx < head_idx_end; head_idx += num_warps)
    {
        // in first iteration, wait for barrier initialization (TODO not needed anymore)
        // afterwards, wait for shared memory reads to be done s.t. we can over-write
        __syncwarp();
        // pre-load gathered_stats data for current head into shared memory
        for (int cp_idx = lane_idx; cp_idx < cp_size; cp_idx += WARP_SIZE)
        {
            int64_t base_offset = cp_idx * (int64_t(num_tokens) * int64_t(num_heads))
                + int64_t(token_idx) * int64_t(num_heads) + int64_t(head_idx);
            memcpy_async_ldgsts<sizeof(float2)>(&smem_stats[warp_idx * cp_size + cp_idx], &gathered_stats[base_offset]);
        }
        asm volatile("cp.async.commit_group;" ::: "memory");

        // Pre-load gathered_o data for current head into shared memory
        static constexpr int O_ELEMENTS = 16 / sizeof(T); // 8 elements per load

        // Each thread loads multiple 16-byte chunks
        for (int cp_idx = 0; cp_idx < cp_size; ++cp_idx)
        {
            int64_t base_offset = cp_idx * (int64_t(num_tokens) * int64_t(num_heads) * int64_t(kv_lora_rank))
                + int64_t(token_idx) * int64_t(num_heads * kv_lora_rank) + int64_t(head_idx) * int64_t(kv_lora_rank);

            for (int v_start = lane_idx * O_ELEMENTS; v_start < kv_lora_rank; v_start += WARP_SIZE * O_ELEMENTS)
            {
                memcpy_async_ldgsts(&smem_o[warp_idx * cp_size * kv_lora_rank + cp_idx * kv_lora_rank + v_start],
                    &gathered_o[base_offset + v_start]);
            }
        }
        asm volatile("cp.async.commit_group;" ::: "memory");

        // wait for the gathered_stats values to be loaded into shared memory
        asm volatile("cp.async.wait_group 1;" ::: "memory");
        float max_values[MAX_VAL_PER_THREAD];
        float sum_values[MAX_VAL_PER_THREAD];
        for (int i = 0; i < MAX_VAL_PER_THREAD; ++i)
        {
            int cp_idx = lane_idx + i * WARP_SIZE;
            float2 stats = cp_idx < cp_size ? smem_stats[warp_idx * cp_size + cp_idx] : make_float2(-INFINITY, 0.F);
            max_values[i] = stats.x;
            sum_values[i] = stats.y;
        }
        float corrected_values[MAX_VAL_PER_THREAD];
        warpReduceCorrectedSum(corrected_values, max_values, sum_values, scale);

        // wait for the gathered_o values to be loaded into shared memory
        asm volatile("cp.async.wait_group 0;" ::: "memory");

        for (int v = lane_idx; v < kv_lora_rank; v += WARP_SIZE)
        {
            float acc = 0.F;
            for (int cp_idx = 0; cp_idx < cp_size; ++cp_idx)
            {
                // correction = corrected_max_exp / global_sum
                float correction = __shfl_sync(0xffffffff, corrected_values[cp_idx / WARP_SIZE], cp_idx % WARP_SIZE);
                // Use shared memory data instead of global memory
                float o_elem
                    = static_cast<float>(smem_o[warp_idx * cp_size * kv_lora_rank + cp_idx * kv_lora_rank + v]);
                acc += o_elem * correction;
            }
            // Store to output: [num_tokens, num_heads * kv_lora_rank]
            int64_t out_offset
                = int64_t(token_idx) * int64_t(num_heads * kv_lora_rank) + int64_t(head_idx * kv_lora_rank + v);
            output[out_offset] = static_cast<T>(acc);
        }
    }
}

template <typename T>
void helixPostProcess(HelixPostProcParams<T> const& params, cudaStream_t stream)
{
    // Check that gathered_o is 16-byte aligned
    TLLM_CHECK_WITH_INFO(reinterpret_cast<uintptr_t>(params.gathered_o) % 16 == 0,
        "gathered_o must be 16-byte aligned for async memcpy");
    // Check that kv_lora_rank * sizeof(T) is a multiple of 16
    TLLM_CHECK_WITH_INFO((params.kv_lora_rank * sizeof(T)) % 16 == 0,
        "kv_lora_rank * sizeof(T) must be a multiple of 16 for async memcpy");
    // Check that cp_size is not larger than the max fallback CP size
    TLLM_CHECK_WITH_INFO(params.cp_size <= FB_MAX_CP, "cp_size > fallback max CP size");
    // If the number of tokens and heads is large enough, we always use the fallback
    bool use_fallback = params.num_tokens * params.num_heads > 256;
    int device, max_shared_mem_per_block, n_sms;
    TLLM_CUDA_CHECK(cudaGetDevice(&device));
    TLLM_CUDA_CHECK(cudaDeviceGetAttribute(&max_shared_mem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    TLLM_CUDA_CHECK(cudaDeviceGetAttribute(&n_sms, cudaDevAttrMultiProcessorCount, device));

    // Choose block and grid sizes
    // Each block processes one token
    int factor = 1;
    while (params.num_tokens * factor < n_sms && params.num_heads % factor == 0)
    {
        factor *= 2;
    }
    dim3 blocks(params.num_tokens, factor);
    int threads = max(MIN_NUM_THREADS, MAX_NUM_THREADS / factor);
    int num_warps = threads / WARP_SIZE;

    size_t shmem_size
        = (sizeof(T) * num_warps * params.cp_size * params.kv_lora_rank + sizeof(float2) * num_warps * params.cp_size);
    use_fallback = use_fallback || max_shared_mem_per_block < shmem_size;
    if (use_fallback)
    {
        threads = WARP_SIZE + params.kv_lora_rank * sizeof(T) / 16;
        dim3 grid(params.num_tokens, params.num_heads);
        helix_postprocess_kernel_fallback<<<grid, threads, 0, stream>>>(
            params.output, params.gathered_o, params.gathered_stats, params.cp_size, params.kv_lora_rank, params.scale);
        return;
    }
    if (shmem_size > 48 * 1024)
    {
        // Set kernel attribute for more dynamic shared memory
        TLLM_CUDA_CHECK(
            cudaFuncSetAttribute(helix_postprocess_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size));
    }
    helix_postprocess_kernel<<<blocks, threads, shmem_size, stream>>>(params.output, params.gathered_o,
        params.gathered_stats, params.cp_size, params.num_tokens, params.num_heads, params.num_heads / factor,
        params.kv_lora_rank, params.scale);
}

#define INSTANTIATE_POST_PROC(T)                                                                                       \
    template void helixPostProcess<T>(HelixPostProcParams<T> const& params, cudaStream_t stream);

INSTANTIATE_POST_PROC(__half);
INSTANTIATE_POST_PROC(__nv_bfloat16);

} // namespace kernels
} // namespace tensorrt_llm
