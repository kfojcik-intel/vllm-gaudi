import torch
from vllm_gaudi.v1.worker.hpu_model_runner import HPUModelRunner
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheTensor,
    KVCacheGroupSpec,
    MambaSpec,
    FullAttentionSpec
)
from vllm.attention.layer import Attention
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.config import VllmConfig, CacheConfig, ModelConfig, ParallelConfig, SchedulerConfig
from vllm_gaudi.extension.environment import _VLLM_VALUES


# Fake Mamba layer
class FakeMamba(MambaBase):
    def __init__(self, shapes):
        self._shapes = shapes

    def get_state_shape(self):
        return self._shapes

    @property
    def mamba_type(self):
        return "mamba2"

    def get_state_dtype(self):
        return (torch.bfloat16, torch.bfloat16)

    def get_attn_backend(self):
        # Return a fake object with .full_cls_name() so HPUModelRunner works
        class BackendMock:
            def full_cls_name(self):
                return "FakeMambaBackend"
        return BackendMock()

# VLLM config
def get_granite_vllm_config():
    scheduler_config = SchedulerConfig(
        max_num_seqs=4,
        max_num_batched_tokens=512,
        max_model_len=131072,
        is_encoder_decoder=False
    )

    model_config = ModelConfig(
        model="ibm-granite/granite-4.0-h-small",
        task="generate",
        tokenizer="ibm-granite/granite-4.0-h-small",
        tokenizer_mode="auto",
        trust_remote_code=True,
        dtype="bfloat16",
        seed=42,
    )

    # ---- mocks for HPU runner ----
    model_config.model_type = "granitemoehybrid"

    class HFConfigMock:
        model_type = "granitemoehybrid"
        hidden_size = 3328
        mamba_expand = 3
        mamba_n_groups = 1  # only 1 group for simplicity
        mamba_n_heads = 48
        mamba_d_head = 64
        mamba_d_state = 48 * 64
        mamba_d_conv = 128
        num_kv_heads = 4
        num_layers = 6
        intermediate_size = hidden_size * mamba_expand

        def get_text_config(self):
            return self

    model_config.hf_config = HFConfigMock()

    cache_config = CacheConfig(
        block_size=128,
        gpu_memory_utilization=0.9,
        swap_space=0,
        cache_dtype="auto",
    )

    parallel_config = ParallelConfig()

    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        scheduler_config=scheduler_config,
        parallel_config=parallel_config,
    )
    return vllm_config


# Layer names
mamba_layers = [
    'model.layers.0.mixer',
    'model.layers.1.mixer',
    'model.layers.2.mixer',
    'model.layers.3.mixer',
    'model.layers.4.mixer'
]
attn_layers = ['model.layers.5.self_attn.attn']

# KV cache specs
mamba_spec = MambaSpec(
    block_size=1024,
    shapes=((3, 3328), (48, 64, 128)),
    dtypes=(torch.bfloat16, torch.bfloat16),
    page_size_padded=819200,
    mamba_type="mamba2",
    num_speculative_blocks=0
)

attn_spec = FullAttentionSpec(
    block_size=400,
    num_kv_heads=4,
    head_size=128,
    dtype=torch.bfloat16,
    sliding_window=None,
    attention_chunk_size=None
)

# KVCache tensors
kv_cache_tensor = KVCacheTensor(
    size=mamba_spec.page_size_padded * 21814,
    shared_by=mamba_layers + attn_layers
)

# KVCache groups
kv_cache_groups = [
    KVCacheGroupSpec(layer_names=mamba_layers, kv_cache_spec=mamba_spec),
    KVCacheGroupSpec(layer_names=attn_layers, kv_cache_spec=attn_spec)
]

kv_cache_config = KVCacheConfig(
    num_blocks=21814,
    kv_cache_tensors=[kv_cache_tensor],
    kv_cache_groups=kv_cache_groups
)

# Initialize HPUModelRunner
DEVICE = "hpu"

# minimal environment for HPU
_VLLM_VALUES['model_type'] = "granitemoehybrid"
_VLLM_VALUES['prompt_attn_impl'] = "fsdpa_impl"

vllm_config = get_granite_vllm_config()

# populate static_forward_context
vllm_config.compilation_config.static_forward_context = {
    'model.layers.0.mixer': FakeMamba(mamba_spec.shapes),
    'model.layers.1.mixer': FakeMamba(mamba_spec.shapes),
    'model.layers.2.mixer': FakeMamba(mamba_spec.shapes),
    'model.layers.3.mixer': FakeMamba(mamba_spec.shapes),
    'model.layers.4.mixer': FakeMamba(mamba_spec.shapes),
    'model.layers.5.self_attn.attn': Attention(
        num_heads=attn_spec.num_kv_heads,
        head_size=attn_spec.head_size,
        scale=1.0 / (attn_spec.head_size ** 0.5),
    ),
}

runner = HPUModelRunner(vllm_config, DEVICE)

# Initialize KV cache
runner.initialize_kv_cache(kv_cache_config)
print("Hybrid KV cache initialized successfully with FakeMamba + Attention layers")


for tensor in runner.kv_cache_config.kv_cache_tensors:
    print(f"KVCacheTensor size: {tensor.size}")
    print(f"Shared by layers: {tensor.shared_by}")
