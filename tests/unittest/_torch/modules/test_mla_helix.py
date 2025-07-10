import pickle
import sys
import time
import traceback
import weakref
from dataclasses import dataclass

import cloudpickle
import pytest
import torch
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

import tensorrt_llm
from tensorrt_llm._torch.attention_backend.interface import (
    KVCacheParams, PositionalEmbeddingParams, RopeParams)
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.attention import MLA
from tensorrt_llm._torch.pyexecutor.llm_request import (LlmRequest,
                                                        LlmRequestState,
                                                        SamplingConfig)
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._torch.utils import model_extra_attrs
from tensorrt_llm._utils import str_dtype_to_binding, torch_dtype_to_str
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.mapping import CpType, Mapping
from tensorrt_llm.sampling_params import SamplingParams

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)


# values for deepseek_v3_lite
@dataclass(kw_only=True, frozen=True)
class Scenario:
    dtype: torch.dtype = torch.bfloat16
    kv_cache_dtype: torch.dtype = torch.bfloat16
    num_layers: int = 1
    num_heads: int = 32
    num_kv_heads: int = 32
    q_lora_rank: int = None
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    hidden_size: int = 2560
    rope_theta: float = 10000.0
    rope_scaling: bool = False
    rope_beta_fast: int = 32
    rope_beta_slow: int = 1
    rope_factor: float = 40.0
    rope_mscale: float = 1.0
    rope_mscale_all_dim: float = 1.0
    rope_original_max_position_embeddings: int = 4096
    rope_type: str = "yarn"
    model_type: str = "deepseek_v3"
    kv_cache_tokens_per_block: int = 64
    predicted_tokens_per_seq: int = 1
    bias: bool = False
    batch: int = 8
    ctx_len: int = 1024
    ref_steps: int = 4

    @property
    def max_position_embeddings(self) -> int:
        # ensure that max_position_embeddings is set large enough for every scenario
        return self.ctx_len + 1


all_scenarios = [
    Scenario(batch=1, ctx_len=1024),
    Scenario(batch=1, ctx_len=2048),
    Scenario(batch=1, ctx_len=4096),
    Scenario(batch=1, ctx_len=8192),
    Scenario(batch=1, ctx_len=16384),
    Scenario(batch=1, ctx_len=32768),
    Scenario(batch=1, ctx_len=65536),
    Scenario(batch=1, ctx_len=131072),
    Scenario(batch=8, ctx_len=1024),
    Scenario(batch=8, ctx_len=2048),
    Scenario(batch=8, ctx_len=4096),
    Scenario(batch=8, ctx_len=8192),
    Scenario(batch=8, ctx_len=16384),
    Scenario(batch=8, ctx_len=32768),
    Scenario(batch=8, ctx_len=65536),
    Scenario(batch=8, ctx_len=131072),
    Scenario(batch=16, ctx_len=1024),
    Scenario(batch=16, ctx_len=2048),
    Scenario(batch=16, ctx_len=4096),
    Scenario(batch=16, ctx_len=8192),
    Scenario(batch=16, ctx_len=16384),
    Scenario(batch=16, ctx_len=32768),
    Scenario(batch=16, ctx_len=65536),
    # this goes OOM
    # Scenario(batch=16, ctx_len=131072),
]

# limit the number of test scenarios to avoid taking too long
test_scenarios = [
    all_scenarios[1],
    all_scenarios[5],
    all_scenarios[8],
    all_scenarios[12],
    all_scenarios[18],
    all_scenarios[19],
]


# default values from deepseek_v3, but will be overwritten by scenario
@dataclass(kw_only=True, frozen=True)
class RopeConfig:
    hidden_size: int = 7168
    num_attention_heads: int = 128
    rope_scaling: dict = {
        "beta_fast": 32,
        "beta_slow": 1,
        "factor": 40.0,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
        "original_max_position_embeddings": 4096,
        "type": "yarn",
    },
    max_position_embeddings: int = 163840
    rope_theta: float = 10000.0
    qk_rope_head_dim: int = 64
    model_type: str = "deepseek_v3"


def _setup_kv_and_metadata(scenario: Scenario, mapping: Mapping,
                           gen_steps: int):
    # Set up KVCacheManager and attn_metadata for MLA
    n_gpu = mapping.world_size
    assert scenario.ctx_len % n_gpu == 0
    ctx_len_per_gpu = scenario.ctx_len // n_gpu
    max_tokens = (
        ctx_len_per_gpu + gen_steps + scenario.kv_cache_tokens_per_block - 1
    ) // scenario.kv_cache_tokens_per_block * scenario.kv_cache_tokens_per_block * scenario.batch
    kv_cache_manager = KVCacheManager(
        KvCacheConfig(
            max_tokens=max_tokens,
            enable_block_reuse=False,
        ),
        # Use SELFKONLY for MLA
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY,
        num_layers=scenario.num_layers,
        num_kv_heads=1,
        head_dim=scenario.kv_lora_rank + scenario.qk_rope_head_dim,
        tokens_per_block=scenario.
        kv_cache_tokens_per_block,  # for test, just use seq_len
        max_seq_len=ctx_len_per_gpu + gen_steps,
        max_batch_size=scenario.batch,
        mapping=mapping,
        dtype=str_dtype_to_binding(torch_dtype_to_str(scenario.kv_cache_dtype)),
    )
    for req_id in range(scenario.batch):
        req = LlmRequest(
            request_id=req_id,
            max_new_tokens=1,
            input_tokens=[1] *
            ctx_len_per_gpu,  # all requests have the same length here
            sampling_config=SamplingConfig(
                SamplingParams()._get_sampling_config()),
            is_streaming=False,
        )
        req.is_dummy_request = True
        req.paged_kv_block_ids = []
        beam_width = 1
        kv_cache_manager.impl.add_sequence(req_id, ctx_len_per_gpu, beam_width,
                                           req)
        req.state = LlmRequestState.GENERATION_IN_PROGRESS
        req.prompt_len = ctx_len_per_gpu
        req.py_prompt_len = req.prompt_len
    attn_metadata = get_attention_backend("TRTLLM").Metadata(
        seq_lens=torch.tensor([ctx_len_per_gpu] * scenario.batch,
                              dtype=torch.int),
        request_ids=list(range(scenario.batch)),
        max_num_requests=scenario.batch,
        num_contexts=scenario.batch,
        prompt_lens=[ctx_len_per_gpu] * scenario.batch,
        max_num_tokens=ctx_len_per_gpu,
        kv_cache_manager=kv_cache_manager,
        kv_cache_params=KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=[0 for _ in range(scenario.batch)],
        ),
        mapping=mapping,
    )
    attn_metadata.prepare()
    return kv_cache_manager, attn_metadata


def _generate_random_weights(mla: MLA):
    # Helper to init a tensor
    def init_uniform(tensor, a=-1.0, b=1.0):
        if tensor is not None:
            torch.nn.init.uniform_(tensor, a=a, b=b)

    def init_block_scale(tensor, orig_tensor):
        if tensor is None or orig_tensor is None:
            return
        x = orig_tensor.view(*orig_tensor.shape[:-2],
                             orig_tensor.shape[-2] // 128, 128,
                             orig_tensor.shape[-1] // 128, 128)
        scale = x.abs().amax(dim=(-3, -1)) / 448.
        tensor.fill_(scale)

    # Linear modules
    for name in ["fused_a", "kv_b_proj", "o_proj"]:
        mod = getattr(mla, name, None)
        if mod is not None:
            init_uniform(mod.weight)
            if hasattr(mod, "bias"):
                init_uniform(mod.bias)

    if hasattr(mla, "v_b_proj"):
        init_uniform(mla.v_b_proj)

    # RMSNorm modules
    for name in ["kv_a_layernorm", "q_a_layernorm"]:
        mod = getattr(mla, name, None)
        if mod is not None and hasattr(mod, "weight"):
            init_uniform(mod.weight, a=0.9, b=1.1)

    # q_b_proj and q_proj (q_proj only in lite mode, aliased as q_b_proj)
    for name in ["q_b_proj", "q_proj"]:
        mod = getattr(mla, name, None)
        if mod is not None:
            init_uniform(mod.weight)
            if hasattr(mod, "bias"):
                init_uniform(mod.bias)

    # k_b_proj_trans (created in create_weights)
    if hasattr(mla, "k_b_proj_trans"):
        # init_uniform(mla.k_b_proj_trans)
        mla.k_b_proj_trans.zero_()
        mla.k_b_proj_trans[:, :, 0] = 1.0
    # k_b_proj_trans_scale (optional)
    if hasattr(mla, "k_b_proj_trans_scale"):
        init_block_scale(mla.k_b_proj_trans_scale, mla.k_b_proj_trans)
    # v_b_proj_scale (optional)
    if hasattr(mla, "v_b_proj_scale"):
        init_block_scale(mla.v_b_proj_scale, mla.v_b_proj)


def _copy_to_cp(weights, param_name, dim, rank, world_size):
    w_dim_per_rank = weights[param_name].shape[dim] // world_size
    w_dim_start = rank * w_dim_per_rank
    w_dim_end = w_dim_start + w_dim_per_rank
    slices = [slice(None)] * weights[param_name].ndim
    slices[dim] = slice(w_dim_start, w_dim_end)
    weights[param_name] = weights[param_name][slices]


def _run_mla_distributed(rank: int, world_size: int, scenario: Scenario,
                         mapping: Mapping, test_params: tuple,
                         ref_output: torch.Tensor, gen_steps: int):
    input_ctx, position_ids_ctx, weights, pos_embd_params = test_params
    # we start with the same position_ids as the reference MLA.
    position_ids_gen = torch.full((scenario.batch, ),
                                  scenario.ctx_len,
                                  dtype=torch.int,
                                  device="cuda")
    extra_attrs = dict()
    config = ModelConfig(mapping=mapping)
    config.extra_attrs = extra_attrs
    mla = MLA(
        hidden_size=scenario.hidden_size,
        num_attention_heads=scenario.num_heads,
        num_key_value_heads=scenario.num_kv_heads,
        qk_nope_head_dim=scenario.qk_nope_head_dim,
        qk_rope_head_dim=scenario.qk_rope_head_dim,
        v_head_dim=scenario.v_head_dim,
        q_lora_rank=scenario.q_lora_rank,
        kv_lora_rank=scenario.kv_lora_rank,
        predicted_tokens_per_seq=scenario.predicted_tokens_per_seq,
        max_position_embeddings=scenario.max_position_embeddings,
        bias=scenario.bias,
        pos_embd_params=pos_embd_params,
        layer_idx=0,
        dtype=scenario.dtype,
        config=config,
    ).cuda()
    # above should have the same config as the reference MLA except for the mapping
    # we update the weights accordingly and should be able to load them
    _copy_to_cp(weights, "o_proj.weight", 1, rank, world_size)
    _copy_to_cp(weights, "v_b_proj", 0, rank, world_size)
    mla.load_state_dict(weights)
    # Set up KVCacheManager and attn_metadata for distributed
    kv_cache_manager, attn_metadata = _setup_kv_and_metadata(
        scenario, mapping, gen_steps)
    extra_attrs["attention_metadata"] = weakref.ref(attn_metadata)
    ctx_len_per_gpu = scenario.ctx_len // world_size
    input_ctx_bs = input_ctx.view(scenario.batch, scenario.ctx_len,
                                  scenario.hidden_size)
    input_gen = input_ctx_bs[:, 0, :].contiguous()

    # split inputs into chunks for each rank
    input_ctx_bs_rank = input_ctx_bs[:, rank * ctx_len_per_gpu:(rank + 1) *
                                     ctx_len_per_gpu, :]
    input_ctx_rank = input_ctx_bs_rank.reshape(
        scenario.batch * ctx_len_per_gpu, scenario.hidden_size).contiguous()
    position_ids_ctx_bs = position_ids_ctx.view(scenario.batch,
                                                scenario.ctx_len)
    position_ids_ctx_bs_rank = position_ids_ctx_bs[:, rank *
                                                   ctx_len_per_gpu:(rank + 1) *
                                                   ctx_len_per_gpu]
    position_ids_ctx_rank = position_ids_ctx_bs_rank.reshape(
        scenario.batch * ctx_len_per_gpu).contiguous()
    # this represents the context step
    with model_extra_attrs(extra_attrs):
        mla(position_ids_ctx_rank, input_ctx_rank, attn_metadata)
    outputs = []
    start = time.time()
    for step in range(gen_steps):
        for req_id in range(scenario.batch):
            kv_cache_manager.impl.add_token(req_id)
        cache_add = step if rank == world_size - 1 else 0
        cached_tokens_per_seq = [
            ctx_len_per_gpu + cache_add for _ in range(scenario.batch)
        ]
        attn_metadata = get_attention_backend("TRTLLM").Metadata(
            seq_lens=torch.tensor([1] * scenario.batch, dtype=torch.int),
            request_ids=list(range(scenario.batch)),
            max_num_requests=scenario.batch,
            num_contexts=0,
            prompt_lens=[ctx_len_per_gpu] * scenario.batch,
            max_num_tokens=ctx_len_per_gpu,
            kv_cache_manager=kv_cache_manager,
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=cached_tokens_per_seq,
            ),
            enable_paged_context_mla=True,
        )
        attn_metadata.prepare()
        extra_attrs["attention_metadata"] = weakref.ref(attn_metadata)
        # note: we don't split position_ids_gen per rank because it is used for
        # RoPE which should be applied in the same way for all ranks !
        with model_extra_attrs(extra_attrs):
            result = mla(position_ids_gen, input_gen, attn_metadata)
        # update position_ids_gen
        position_ids_gen += 1
        if step < scenario.ref_steps:
            outputs.append(result)
        elif step == scenario.ref_steps:
            start = time.time()
    end = time.time()
    if gen_steps == scenario.ref_steps:
        avg_gen_time = float('inf')
    else:
        avg_gen_time = (end - start) / (gen_steps - scenario.ref_steps)
    throughput = scenario.batch / avg_gen_time
    print(f"Rank {rank} {world_size}-GPU: time taken for "
          f"{gen_steps - scenario.ref_steps} steps: "
          f"{end - start} s, throughput: {throughput} MLA/s")
    output = torch.stack(outputs, dim=0)
    kv_cache_manager.shutdown()

    # every rank should have the same output and checks against the reference output
    # note: need to use fairly high tolerances because the softmax stats can lose
    # a lot of precision
    atol = 1e-1
    rtol = 1e-2

    err = torch.abs(output - ref_output)
    ref_abs = torch.abs(ref_output)
    rel_err = err / ref_abs
    # always print largest error and its index
    max_err, max_err_idx = torch.max(err, None)
    max_rel_err, max_rel_err_idx = torch.max(rel_err, None)
    print(
        f"Rank {rank} {world_size}-GPU: max abs error: {max_err}, index: {max_err_idx}, max rel error: {max_rel_err}, index: {max_rel_err_idx}"
    )
    isclose = err < atol + rtol * ref_abs
    if not isclose.all().item():
        n_mismatch = (isclose == False).sum().item()
        ratio_mismatch = n_mismatch / output.numel()
        print(
            f"Rank {rank} {world_size}-GPU: {n_mismatch} mismatches, ratio: {ratio_mismatch}"
        )
        # allow up to 7.5% mismatch for now, due to how latent cache is not set correctly for non-last rank
        # TODO: fix this
        assert ratio_mismatch < 0.075


@torch.inference_mode
def _full_test_multi_gpu(rank: int, world_size: int, scenario: Scenario,
                         gen_steps: int):
    if scenario.rope_scaling:
        rope_scaling = {
            "beta_fast": scenario.rope_beta_fast,
            "beta_slow": scenario.rope_beta_slow,
            "factor": scenario.rope_factor,
            "mscale": scenario.rope_mscale,
            "mscale_all_dim": scenario.rope_mscale_all_dim,
            "original_max_position_embeddings":
            scenario.rope_original_max_position_embeddings,
            "type": scenario.rope_type,
        }
    else:
        rope_scaling = None
    rope_config = RopeConfig(
        hidden_size=scenario.hidden_size,
        num_attention_heads=scenario.num_heads,
        rope_scaling=rope_scaling,
        max_position_embeddings=scenario.max_position_embeddings,
        rope_theta=scenario.rope_theta,
        qk_rope_head_dim=scenario.qk_rope_head_dim,
        model_type=scenario.model_type)
    torch.manual_seed(42)
    input_ctx = torch.empty(scenario.batch * scenario.ctx_len,
                            scenario.hidden_size,
                            dtype=scenario.dtype,
                            device="cuda").uniform_(-1, 1)
    input_ctx.zero_()
    input_ctx.view(scenario.batch, scenario.ctx_len,
                   scenario.hidden_size)[:, :, 0] = 1.0
    position_ids_ctx = torch.arange(scenario.ctx_len,
                                    device="cuda").repeat(scenario.batch)
    position_ids_gen = torch.full((scenario.batch, ),
                                  scenario.ctx_len,
                                  dtype=torch.int,
                                  device="cuda")

    pos_embd_params = PositionalEmbeddingParams(
        type=PositionEmbeddingType.yarn,
        rope=RopeParams.from_config(rope_config),
        is_neox=False,
    )

    mla = MLA(
        hidden_size=scenario.hidden_size,
        num_attention_heads=scenario.num_heads,
        num_key_value_heads=scenario.num_kv_heads,
        qk_nope_head_dim=scenario.qk_nope_head_dim,
        qk_rope_head_dim=scenario.qk_rope_head_dim,
        v_head_dim=scenario.v_head_dim,
        q_lora_rank=scenario.q_lora_rank,
        kv_lora_rank=scenario.kv_lora_rank,
        predicted_tokens_per_seq=scenario.predicted_tokens_per_seq,
        max_position_embeddings=scenario.max_position_embeddings,
        bias=scenario.bias,
        pos_embd_params=pos_embd_params,
        layer_idx=0,
        dtype=scenario.dtype,
    ).cuda()
    _generate_random_weights(mla)
    weights = mla.state_dict()
    # up to this point, all ranks should have same tensors because the seed is the same
    # now we run the reference MLA on rank 0
    if rank == 0:
        input_gen = input_ctx.view(scenario.batch, scenario.ctx_len,
                                   scenario.hidden_size)[:, 0, :].contiguous()
        # Reference output (single GPU, but with correct KV/metadata setup)
        ref_mapping = Mapping(world_size=1, tp_size=1, rank=0)
        ref_kv_cache_manager, ref_attn_metadata = _setup_kv_and_metadata(
            scenario, ref_mapping, gen_steps)
        # this represents the context step
        mla(position_ids_ctx, input_ctx, ref_attn_metadata)
        ref_outputs = []
        start = time.time()
        for step in range(gen_steps):
            for req_id in range(scenario.batch):
                ref_kv_cache_manager.impl.add_token(req_id)
            ref_attn_metadata = get_attention_backend("TRTLLM").Metadata(
                seq_lens=torch.tensor([1] * scenario.batch, dtype=torch.int),
                request_ids=list(range(scenario.batch)),
                max_num_requests=scenario.batch,
                num_contexts=0,
                prompt_lens=[scenario.ctx_len] * scenario.batch,
                max_num_tokens=scenario.ctx_len,
                kv_cache_manager=ref_kv_cache_manager,
                kv_cache_params=KVCacheParams(
                    use_cache=True,
                    num_cached_tokens_per_seq=[
                        scenario.ctx_len + step for _ in range(scenario.batch)
                    ],
                ),
                enable_paged_context_mla=True,
            )
            ref_attn_metadata.prepare()
            result = mla(position_ids_gen, input_gen, ref_attn_metadata)
            # update position_ids_gen
            position_ids_gen += 1
            if step < scenario.ref_steps:
                ref_outputs.append(result)
            elif step == scenario.ref_steps:
                start = time.time()
        end = time.time()
        if gen_steps == scenario.ref_steps:
            avg_gen_time = float('inf')
        else:
            avg_gen_time = (end - start) / (gen_steps - scenario.ref_steps)
        throughput = scenario.batch / avg_gen_time
        print(f"Time taken for {gen_steps - scenario.ref_steps} steps: "
              f"{end - start} s, throughput: {throughput} MLA/s")
        ref_output = torch.stack(ref_outputs, dim=0)
        ref_kv_cache_manager.shutdown()
    else:
        ref_output = torch.empty(scenario.ref_steps,
                                 scenario.batch,
                                 scenario.hidden_size,
                                 dtype=scenario.dtype,
                                 device="cuda")

    # Distributed mapping for helix
    mapping = Mapping(world_size=world_size,
                      rank=rank,
                      cp_size=world_size,
                      cp_config={'cp_type': CpType.HELIX})
    # we use cp_allgather here because there is no broadcast op across CP group
    from tensorrt_llm._torch.distributed.ops import cp_allgather
    ref_output_all = cp_allgather(ref_output, mapping=mapping, dim=0)
    # we only need the values from rank 0
    ref_output = ref_output_all.view(world_size, *ref_output.shape)[0]
    test_params = (input_ctx, position_ids_ctx, weights, pos_embd_params)
    _run_mla_distributed(rank, world_size, scenario, mapping, test_params,
                         ref_output, gen_steps)


def _run_single_rank(func, *args, **kwargs):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    print(f"rank {rank} starting")
    try:
        ret = func(rank, *args, **kwargs)
        print(f"rank {rank} done")
        return ret
    except Exception:
        traceback.print_exc()
        tb = traceback.format_exc()
        raise Exception(f"\n\nError occurred. Original traceback is\n{tb}\n")


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="needs 2 GPUs to run this test")
@pytest.mark.parametrize("scenario",
                         test_scenarios,
                         ids=lambda x: f"scenario: {x}")
def test_mla_helix_distributed(scenario: Scenario):
    world_size = 2
    with MPIPoolExecutor(max_workers=world_size) as executor:
        gen_steps = scenario.ref_steps
        results = executor.map(
            _run_single_rank,
            *zip(*[(_full_test_multi_gpu, world_size, scenario, gen_steps)] *
                 world_size))
        for r in results:
            assert r is None


if __name__ == "__main__":
    for scenario in test_scenarios:
        timing_steps = 256
        gen_steps = scenario.ref_steps + timing_steps
        print(f"Running scenario: {scenario} and timing {timing_steps} steps")
        test_mla_helix_distributed(scenario)
