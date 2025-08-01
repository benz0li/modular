# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from collections.abc import Iterable
from itertools import islice

from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    ShardingStrategy,
    TensorType,
    TensorValue,
    TensorValueLike,
    Value,
    ops,
)
from max.graph.ops.allreduce import matmul_allreduce
from max.nn.comm.allreduce import Allreduce

from ..embedding import VocabParallelEmbedding
from ..kv_cache import (
    ContinuousBatchingKVCacheCollection,
    FetchContinuousBatchingKVCacheCollection,
    FetchPagedKVCacheCollection,
    KVCacheParams,
    PagedKVCacheCollection,
)
from ..layer import LayerList, Module, Shardable
from ..linear import ColumnParallelLinear, DistributedGemmConfig
from ..norm import DistributedRMSNorm
from .transformer import ReturnLogits


def take(it: Iterable[Value], n: int) -> list[Value]:
    """Return the next *n* items from *it* as a list."""
    return list(islice(it, n))


# TODO (pavan): clean up duplicate instances of distribute_value, shard_col_value,
# shard_row_value across the codebase into a multi gpu utils file
def distribute_value(v, devices: list[DeviceRef]):  # noqa: ANN001
    return [v.to(device) for device in devices]


class DistributedTransformerBlock(Module):
    """Stack of Attention, FeedForward, and RMSNorm layers."""

    def __init__(
        self,
        attention: Module,
        mlp: Shardable,
        attention_norm: DistributedRMSNorm,
        mlp_norm: DistributedRMSNorm,
        devices: list[DeviceRef],
        distributed_gemm_config: DistributedGemmConfig | None = None,
    ) -> None:
        super().__init__()
        self.self_attn = attention
        self.mlp = mlp
        self.mlp.sharding_strategy = ShardingStrategy.tensor_parallel(
            len(devices)
        )
        self.mlp_shards = mlp.shard(devices)

        self.input_layernorm = attention_norm
        self.post_attention_layernorm = mlp_norm
        self.devices = devices

        self.distributed_gemm_config = distributed_gemm_config
        self.allreduce = Allreduce(num_accelerators=len(devices))

    def __call__(
        self,
        layer_idx: TensorValue,
        xs: list[TensorValue],
        signal_buffers: list[BufferValue],
        kv_collections: list[
            ContinuousBatchingKVCacheCollection | PagedKVCacheCollection
        ],
        input_row_offsets: list[TensorValue],
    ) -> list[TensorValue]:
        attn_outs = self.self_attn(
            layer_idx,
            self.input_layernorm(xs),
            signal_buffers,
            kv_collections,
            input_row_offsets,
        )

        hs = [x + attn_out for x, attn_out in zip(xs, attn_outs)]

        norm_outs = self.post_attention_layernorm(hs)
        mlp_outs = [
            self.mlp_shards[i](norm_outs[i])  # type: ignore[operator]
            for i in range(len(self.mlp_shards))
        ]

        if (
            self.distributed_gemm_config is None
            or not self.distributed_gemm_config.enable_matmul_allreduce
        ):
            mlp_outs = self.allreduce(mlp_outs, signal_buffers)
        else:
            # Special matmul + allreduce split version
            # extract the sharded weights from the last linear layers
            weights = [layer.down_proj.weight for layer in self.mlp_shards]  # type: ignore[attr-defined]
            mlp_outs = matmul_allreduce(
                mlp_outs,
                weights,
                signal_buffers,
            )

        hs = [h + mlp_out for h, mlp_out in zip(hs, mlp_outs)]

        return hs


class DistributedTransformer(Module):
    """Transformer model consisting for TransformerBlock layers."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        layers: list[DistributedTransformerBlock],
        norm: DistributedRMSNorm,
        output: ColumnParallelLinear,
        embedding: VocabParallelEmbedding,
        kv_params: KVCacheParams,
        kv_collection_constructor: (
            FetchContinuousBatchingKVCacheCollection
            | FetchPagedKVCacheCollection
        ),
        devices: list[DeviceRef],
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
        use_subgraphs: bool = False,
        subgraph_layer_groups: list[list[int]] | None = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.layers = LayerList(layers)
        self.norm = norm
        self.lm_head = output
        self.embed_tokens = embedding
        self.kv_params = kv_params
        self.kv_collection_constructor = kv_collection_constructor
        self.return_logits = return_logits
        self.devices = devices
        self.use_subgraphs = use_subgraphs
        if subgraph_layer_groups is None:
            # If no subgraph layer groups are provided, assume that all layers
            # are in a single group.
            subgraph_layer_groups = [[i for i in range(len(layers))]]
        self.subgraph_layer_groups = subgraph_layer_groups
        if self.return_logits == ReturnLogits.VARIABLE:
            raise ValueError(
                "DistributedTransformer does not support variable logits."
            )

    def __call__(
        self,
        tokens: TensorValueLike,
        signal_buffers: list[BufferValue],
        kv_cache_inputs_per_dev: list[tuple[TensorValue, ...]],
        return_n_logits: TensorValue,
        input_row_offsets: TensorValue,
    ) -> tuple[TensorValue, ...]:
        h = self.embed_tokens(tokens, signal_buffers)

        kv_collections = [
            self.kv_collection_constructor(*kv_cache_inputs)
            for kv_cache_inputs in kv_cache_inputs_per_dev
        ]

        input_row_offsets_ = distribute_value(input_row_offsets, self.devices)

        if self.use_subgraphs:
            subgraph_input_types = [
                TensorType(DType.uint32, shape=(), device=DeviceRef.CPU()),
                [hidden.type for hidden in h],
                [signal_buffer.type for signal_buffer in signal_buffers],
                [kv_collection.type for kv_collection in kv_collections],
                [offset.type for offset in input_row_offsets_],
            ]

            # First, we need to build the subgraphs for each layer group.
            subgraphs = []
            for group_idx, layer_group in enumerate(self.subgraph_layer_groups):
                assert len(layer_group) > 0, (
                    "Subgraph layer groups must contain at least one layer"
                )
                subgraph_layer = self.layers[layer_group[0]]
                assert isinstance(
                    subgraph_layer, DistributedTransformerBlock
                ), "Subgraph layer must be a DistributedTransformerBlock"
                subgraphs.append(
                    subgraph_layer.build_subgraph(
                        f"dist_transformer_block_{group_idx}",
                        subgraph_input_types,
                        f"layers.{layer_group[0]}.",
                    )
                )

            # Then, we need to call the subgraphs for each layer group.
            for idx, layer in enumerate(self.layers):
                has_subgraph = False
                for group_idx, layer_group in enumerate(
                    self.subgraph_layer_groups
                ):
                    if idx in layer_group:
                        has_subgraph = True
                        h = [
                            x.tensor
                            for x in ops.call(
                                subgraphs[group_idx],
                                ops.constant(
                                    idx, DType.uint32, device=DeviceRef.CPU()
                                ),
                                *h,
                                *signal_buffers,
                                *kv_collections,
                                *input_row_offsets_,
                                prefix=f"layers.{idx}.",
                            )
                        ]
                        break
                if not has_subgraph:
                    # If no subgraph was found, call the layer directly.
                    h = layer(
                        ops.constant(idx, DType.uint32, device=DeviceRef.CPU()),
                        h,
                        signal_buffers,
                        kv_collections,
                        input_row_offsets_,
                    )
        else:
            for idx, layer in enumerate(self.layers):
                h = layer(
                    ops.constant(idx, DType.uint32, device=DeviceRef.CPU()),
                    h,
                    signal_buffers,
                    kv_collections,
                    input_row_offsets_,
                )
        h0 = h[0]
        last_token_indices = input_row_offsets[1:] - 1
        last_token_h = ops.gather(h0, last_token_indices, axis=0)
        last_token_distributed = distribute_value(last_token_h, self.devices)
        last_logits = ops.cast(
            self.lm_head(self.norm(last_token_distributed), signal_buffers)[0],
            DType.float32,
        )

        logits = None
        offsets = None

        if self.return_logits == ReturnLogits.VARIABLE:
            return_n_logits_range = ops.range(
                return_n_logits[0],
                ops.constant(0, DType.int64, device=DeviceRef.CPU()),
                ops.constant(-1, DType.int64, device=DeviceRef.CPU()),
                out_dim="return_n_logits_range",
                device=self.devices[0],
            )
            offsets = (
                ops.unsqueeze(input_row_offsets[1:], -1) - return_n_logits_range
            )
            last_indices = ops.reshape(offsets, shape=(-1,))
            logits = ops.gather(
                ops.cast(
                    self.lm_head(self.norm(h), signal_buffers)[0], DType.float32
                ),
                last_indices,
                axis=0,
            )
            offsets = ops.range(
                ops.constant(0, DType.int64, device=DeviceRef.CPU()),
                last_indices.shape[0] + return_n_logits[0],
                return_n_logits[0],
                out_dim="logit_offsets",
                device=self.devices[0],
            )
        elif self.return_logits == ReturnLogits.ALL:
            logits = ops.cast(
                self.lm_head(self.norm(h), signal_buffers)[0], DType.float32
            )
            offsets = input_row_offsets

        if logits is not None and offsets is not None:
            return (last_logits, logits, offsets)
        else:
            return (last_logits,)
