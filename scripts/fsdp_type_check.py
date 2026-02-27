import sys, os, functools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from accelerate import Accelerator, FullyShardedDataParallelPlugin
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from modeling.modules.attention import TransformerBlock
import torch

fsdp_plugin = FullyShardedDataParallelPlugin(
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
    auto_wrap_policy=functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={TransformerBlock}),
    use_orig_params=True, cpu_offload=False,
)
acc = Accelerator(fsdp_plugin=fsdp_plugin, mixed_precision='bf16')

net = torch.nn.Linear(8, 8)
prepared = acc.prepare(net)
unwrapped = acc.unwrap_model(prepared)

print(f"[rank {acc.process_index}] type(prepared)  = {type(prepared).__name__}")
print(f"[rank {acc.process_index}] type(unwrapped) = {type(unwrapped).__name__}")
print(f"[rank {acc.process_index}] isinstance(prepared, FSDP)  = {isinstance(prepared, FSDP)}")
print(f"[rank {acc.process_index}] isinstance(unwrapped, FSDP) = {isinstance(unwrapped, FSDP)}")

acc.wait_for_everyone()
print(f"[rank {acc.process_index}] done")
