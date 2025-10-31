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

from math import ceildiv
from sys import size_of

from gpu import barrier
from gpu.host import DeviceContext
from gpu.id import block_idx, thread_idx
from layout import Layout, LayoutTensor
from layout._fillers import random
from layout._utils import ManagedLayoutTensor
from layout.tma_async import SharedMemBarrier
from memory import stack_allocation
from memory.pointer import _GPUAddressSpace
from testing import assert_equal

from utils.index import Index, IndexList
from layout.int_tuple import IntTuple, product, depth, to_index_list
from layout.layout import zipped_divide, blocked_product, print_layout

from random import random_si64
from linalg.arch.sm100._tma import (
    create_tma_descriptor,
    TMALoad,
    copy,
    UInt32Indices,
    to_swizzle,
)
from gpu.memory import AddressSpace
from gpu.host._tensormap import SwizzleMode
from gpu import WARP_SIZE
from testing import assert_equal
from linalg.arch.sm100.mma import max_contiguous_tile_shape, Major


# funtionally equivalent to idx2crd
# but avoids runtime_*
# converts linear index to column-major coordinates
fn calculate_coordinate[
    global_layout: Layout, tile_shape: IntTuple
](linear_index: Int) -> UInt32Indices[global_layout.rank()]:
    alias coalesced_tile = tile_shape.product_flatten()
    alias smem_tiler = Layout(coalesced_tile)

    constrained[global_layout.rank() == smem_tiler.rank(),]()

    alias rank = global_layout.rank()

    alias zipped_layout = zipped_divide(global_layout, smem_tiler)[1]
    alias zipped_shape = zipped_layout.shape

    var indices = UInt32Indices[rank](0)
    var elements_per_dimension = IndexList[rank](0)

    var prev = 1

    @parameter
    for i in range(rank):
        alias index = rank - i - 1
        alias dimension = zipped_shape[index].value()

        elements_per_dimension[index] = prev
        prev = prev * dimension

    var current_coordinate = linear_index
    alias tile_dims = to_index_list[rank, DType.uint32](coalesced_tile)

    @parameter
    for index in range(rank):
        indices[index] = current_coordinate // elements_per_dimension[index]
        current_coordinate = current_coordinate % elements_per_dimension[index]

    return indices * tile_dims


@always_inline
fn shared_to_global_2D[
    OOB_access: Bool
](
    smem_tile: LayoutTensor,
    dst: LayoutTensor,
    tiled_coordinate: IndexList[2, element_type = DType.uint32],
):
    alias smem_dim0 = product(smem_tile.layout.shape[0])
    alias smem_dim1 = product(smem_tile.layout.shape[1])

    alias global_dim0 = product(dst.layout.shape[0])
    alias global_dim1 = product(dst.layout.shape[1])

    if thread_idx.x == 0:

        @parameter
        if OOB_access:
            var end_row = min(global_dim0, smem_dim0 + tiled_coordinate[0])
            var end_col = min(global_dim1, smem_dim1 + tiled_coordinate[1])

            for i in range(tiled_coordinate[0], end_row):
                for j in range(tiled_coordinate[1], end_col):
                    dst[i, j] = rebind[
                        SIMD[dst.dtype, dst.element_layout.size()]
                    ](
                        smem_tile[
                            i - tiled_coordinate[0], j - tiled_coordinate[1]
                        ]
                    )

        else:
            var dst_tile = dst.tile[smem_dim0, smem_dim1](
                tiled_coordinate[0] // product(smem_dim0),
                tiled_coordinate[1] // product(smem_dim1),
            )  # flip because of column-major coordinate

            for i in range(smem_dim0):
                for j in range(smem_dim1):
                    dst_tile[i, j] = rebind[
                        SIMD[dst_tile.dtype, dst_tile.element_layout.size()]
                    ](smem_tile[i, j])


@always_inline
fn shared_to_global_3D[
    OOB_access: Bool
](
    smem_tile: LayoutTensor,
    dst: LayoutTensor,
    tiled_coordinate: IndexList[3, element_type = DType.uint32],
):
    alias smem_dim0 = product(smem_tile.layout.shape[0])
    alias smem_dim1 = product(smem_tile.layout.shape[1])
    alias smem_dim2 = product(smem_tile.layout.shape[2])

    alias global_dim0 = product(dst.layout.shape[0])
    alias global_dim1 = product(dst.layout.shape[1])
    alias global_dim2 = product(dst.layout.shape[2])

    if thread_idx.x == 0:

        @parameter
        if OOB_access:
            var end_block = min(global_dim0, smem_dim0 + tiled_coordinate[0])
            var end_row = min(global_dim1, smem_dim1 + tiled_coordinate[1])
            var end_col = min(global_dim2, smem_dim2 + tiled_coordinate[2])

            for b in range(tiled_coordinate[0], end_block):
                for i in range(tiled_coordinate[1], end_row):
                    for j in range(tiled_coordinate[2], end_col):
                        dst[b, i, j] = rebind[
                            SIMD[dst.dtype, dst.element_layout.size()]
                        ](
                            smem_tile[
                                b - tiled_coordinate[0],
                                i - tiled_coordinate[1],
                                j - tiled_coordinate[2],
                            ]
                        )
        else:
            var dst_tile = dst.tile[smem_dim0, smem_dim1, smem_dim2](
                tiled_coordinate[0] // product(smem_dim0),
                tiled_coordinate[1] // product(smem_dim1),
                tiled_coordinate[2] // product(smem_dim2),
            )  # flip because of column-major coordinate

            for i in range(smem_dim0):
                for j in range(smem_dim1):
                    for k in range(smem_dim2):
                        dst_tile[i, j, k] = rebind[
                            SIMD[dst_tile.dtype, dst_tile.element_layout.size()]
                        ](smem_tile[i, j, k])


# Test loading a single 2d tile.
@__llvm_arg_metadata(load_policy, `nvvm.grid_constant`)
fn test_tma_load_kernel[
    dtype: DType,
    global_layout: Layout,
    smem_layout: Layout,
    tile_shape: IntTuple,
    swizzle_mode: SwizzleMode,
    OOB_access: Bool = False,
](
    dst: LayoutTensor[dtype, global_layout, MutableAnyOrigin],
    load_policy: TMALoad[dtype, tile_shape, swizzle_mode],
):
    alias expected_bytes = smem_layout.size() * size_of[dtype]()

    var smem_tile = LayoutTensor[
        dtype,
        smem_layout,
        MutableAnyOrigin,
        address_space = _GPUAddressSpace.SHARED,
        alignment=128,
    ].stack_allocation()

    var mbar_ptr = stack_allocation[
        1,
        SharedMemBarrier,
        address_space = _GPUAddressSpace.SHARED,
        alignment=8,
    ]()

    var linear_index = Int(block_idx.x)

    var coordinate = calculate_coordinate[global_layout, smem_layout.shape](
        linear_index
    )

    if thread_idx.x == 0:
        mbar_ptr[0].init()
        mbar_ptr[0].expect_bytes(expected_bytes)
        copy(
            load_policy,
            smem_tile,
            mbar_ptr,
            coordinate,
        )
    # Ensure all threads sees initialized mbarrier
    barrier()

    mbar_ptr[0].wait()

    alias smem_indices = to_index_list[coordinate.size, DType.uint32](
        smem_layout.shape
    )
    var tiled_coordinate = coordinate

    @parameter
    if coordinate.size == 2:
        shared_to_global_2D[OOB_access](
            smem_tile, dst, rebind[UInt32Indices[2]](tiled_coordinate)
        )

    elif coordinate.size == 3:
        shared_to_global_3D[OOB_access](
            smem_tile, dst, rebind[UInt32Indices[3]](tiled_coordinate)
        )


def test_2D_swizzle[
    swizzle_mode: SwizzleMode,
    global_shape: IntTuple,
    load_shape: IntTuple,
](reference_tensor: LayoutTensor, result_tensor: LayoutTensor,) -> Int:
    var total_errors = 0

    alias load_shape_m = product(load_shape[0])
    alias load_shape_n = product(load_shape[1])

    alias m_tiles = product(global_shape[0]) // load_shape_m
    alias n_tiles = product(global_shape[1]) // load_shape_n

    alias swizzle = to_swizzle[result_tensor.dtype, swizzle_mode]()

    for i in range(m_tiles):
        for j in range(n_tiles):
            var reference_tile = reference_tensor.tile[
                load_shape_m, load_shape_n
            ](i, j)

            var result_tile = result_tensor.tile[load_shape_m, load_shape_n](
                i, j
            )

            for ii in range(load_shape_m):
                for jj in range(load_shape_n):
                    offset = swizzle(ii * load_shape_n + jj)

                    m_offset = offset // load_shape_n
                    n_offset = offset % load_shape_n

                    if reference_tile[ii, jj] != rebind[
                        SIMD[
                            reference_tensor.dtype,
                            reference_tensor.element_layout.size(),
                        ]
                    ](result_tile[m_offset, n_offset]):
                        total_errors += 1

    return total_errors


def test_3D_swizzle[
    swizzle_mode: SwizzleMode,
    global_shape: IntTuple,
    load_shape: IntTuple,
](reference_tensor: LayoutTensor, result_tensor: LayoutTensor,) -> Int:
    alias load_shape_b = product(load_shape[0])
    alias load_shape_m = product(load_shape[1])
    alias load_shape_n = product(load_shape[2])

    alias b_tiles = product(global_shape[0]) // load_shape_b
    alias m_tiles = product(global_shape[1]) // load_shape_m
    alias n_tiles = product(global_shape[2]) // load_shape_n

    alias swizzle = to_swizzle[result_tensor.dtype, swizzle_mode]()

    var total_errors = 0

    for b in range(b_tiles):
        for i in range(m_tiles):
            for j in range(n_tiles):
                var reference_tile = reference_tensor.tile[
                    load_shape_b, load_shape_m, load_shape_n
                ](b, i, j)

                var result_tile = result_tensor.tile[
                    load_shape_b, load_shape_m, load_shape_n
                ](b, i, j)

                for bb in range(load_shape_b):
                    for ii in range(load_shape_m):
                        for jj in range(load_shape_n):
                            var offset = swizzle(
                                (bb * load_shape_m * load_shape_n)
                                + (ii * load_shape_n + jj)
                            )

                            b_offset = offset // (load_shape_m * load_shape_n)
                            offset = offset % (load_shape_m * load_shape_n)

                            m_offset = offset // load_shape_n
                            n_offset = offset % load_shape_n

                            if reference_tile[bb, ii, jj] != rebind[
                                SIMD[
                                    reference_tensor.dtype,
                                    reference_tensor.element_layout.size(),
                                ]
                            ](result_tile[b_offset, m_offset, n_offset]):
                                total_errors += 1

    return total_errors


def test_tma_load[
    global_shape: IntTuple,
    smem_layout: Layout,
    load_shape: IntTuple,
    dtype: DType,
    swizzle_mode: SwizzleMode = SwizzleMode.NONE,
    OOB_access: Bool = False,
](ctx: DeviceContext):
    constrained[
        depth(global_shape) == depth(load_shape) and depth(load_shape) == 1,
        "Global shape and SMEM shape must have the same depth",
    ]()

    alias total_global_elements = product(global_shape)
    alias total_smem_elements = smem_layout.size()
    alias total_load_elements = product(load_shape)

    constrained[
        total_smem_elements % total_load_elements == 0,
        "shared memory shape must be divisble by load shape",
    ]()

    alias total_tiles = ceildiv(total_global_elements, total_smem_elements)

    var global_buffer_host_reference = ctx.enqueue_create_host_buffer[dtype](
        total_global_elements
    )
    var global_buffer_host_result = ctx.enqueue_create_host_buffer[dtype](
        total_global_elements
    )

    ctx.synchronize()

    var global_buffer_src_tensor = ManagedLayoutTensor[
        dtype, Layout.row_major(global_shape)
    ](ctx)

    var global_buffer_dst_tensor = ManagedLayoutTensor[
        dtype, Layout.row_major(global_shape)
    ](ctx)

    for i in range(total_global_elements):
        global_buffer_host_reference[i] = random_si64(0, 20).cast[dtype]()

    ctx.enqueue_copy(
        global_buffer_src_tensor.tensor().ptr, global_buffer_host_reference
    )

    var descriptor = create_tma_descriptor[
        dtype, load_shape, swizzle_mode=swizzle_mode
    ](global_buffer_src_tensor.device_tensor(), ctx)

    var load_policy = TMALoad(descriptor)

    alias kernel = test_tma_load_kernel[
        dtype,
        global_buffer_dst_tensor.layout,
        smem_layout,
        type_of(load_policy).tile_shape,
        swizzle_mode,
        OOB_access=OOB_access,
    ]

    ctx.enqueue_function[kernel](
        global_buffer_dst_tensor.device_tensor(),
        load_policy,
        grid_dim=(total_tiles),
        block_dim=(WARP_SIZE),
    )

    ctx.enqueue_copy(
        global_buffer_host_result, global_buffer_dst_tensor.tensor().ptr
    )

    ctx.synchronize()

    var total_errors = 0

    @parameter
    if swizzle_mode == SwizzleMode.NONE:
        for i in range(total_global_elements):
            if global_buffer_host_reference[i] != global_buffer_host_result[i]:
                total_errors += 1

        assert_equal(total_errors, 0)
    else:
        alias GlobalTensorType = LayoutTensor[
            dtype, Layout.row_major(global_shape), MutableAnyOrigin
        ]

        var reference_tensor = GlobalTensorType(global_buffer_host_reference)
        var result_tensor = GlobalTensorType(global_buffer_host_result)

        @parameter
        if len(global_shape) == 2:
            total_errors += test_2D_swizzle[
                swizzle_mode, global_shape, load_shape
            ](reference_tensor, result_tensor)

        else:
            total_errors += test_3D_swizzle[
                swizzle_mode, global_shape, load_shape
            ](reference_tensor, result_tensor)

        assert_equal(total_errors, 0)


def main():
    with DeviceContext() as ctx:
        print("Test TMA horizontal loads")

        alias desc_shape = IntTuple(2, 64)

        test_tma_load[
            IntTuple(8, 512),
            TMALoad[DType.bfloat16, desc_shape].get_2D_smem_layout[2, 4](),
            desc_shape,
            DType.bfloat16,
        ](ctx)

        print("Test TMA No Swizzle")

        test_tma_load[
            IntTuple(64, 64),
            Layout.row_major(32, 32),
            IntTuple(32, 32),
            DType.bfloat16,
        ](ctx)

        test_tma_load[
            IntTuple(4, 128, 128),
            Layout.row_major(2, 128, 64),
            IntTuple(2, 128, 64),
            DType.bfloat16,
        ](ctx)

        test_tma_load[
            IntTuple(4, 128, 128),
            Layout.row_major(2, 128, 64),
            IntTuple(1, 64, 64),
            DType.bfloat16,
        ](ctx)

        alias load_tile_shape[
            swizzle_mode: SwizzleMode, major: Major
        ] = max_contiguous_tile_shape[
            DType.bfloat16,
            IndexList[2](128, 128),
            major=major,
            swizzle_mode=swizzle_mode,
        ]()

        alias load_tile_shape_32B_K = load_tile_shape[SwizzleMode._32B, Major.K]
        alias load_tile_shape_64B_K = load_tile_shape[SwizzleMode._64B, Major.K]
        alias load_tile_shape_128B_K = load_tile_shape[
            SwizzleMode._128B, Major.K
        ]
        alias load_tile_shape_128B_MN = load_tile_shape[
            SwizzleMode._128B, Major.MN
        ]

        alias gmem_shape = IntTuple(512, 128)
        alias smem_layout[
            swizzle_mode: SwizzleMode, major: Major
        ] = Layout.row_major(
            128 * 2, load_tile_shape[swizzle_mode, major][1].value()
        )

        print("TMA swizzle k-major")

        test_tma_load[
            gmem_shape,
            smem_layout[SwizzleMode._32B, Major.K],
            load_tile_shape_32B_K,
            DType.bfloat16,
            swizzle_mode = SwizzleMode._32B,
        ](ctx)

        test_tma_load[
            gmem_shape,
            smem_layout[SwizzleMode._64B, Major.K],
            load_tile_shape_64B_K,
            DType.bfloat16,
            swizzle_mode = SwizzleMode._64B,
        ](ctx)

        test_tma_load[
            gmem_shape,
            smem_layout[SwizzleMode._128B, Major.K],
            load_tile_shape_128B_K,
            DType.bfloat16,
            swizzle_mode = SwizzleMode._128B,
        ](ctx)

        print("TMA swizzle mn-major")

        test_tma_load[
            gmem_shape,
            smem_layout[SwizzleMode._128B, Major.K],
            load_tile_shape_128B_MN,
            DType.bfloat16,
            swizzle_mode = SwizzleMode._128B,
        ](ctx)

        print("TMA OOB access")

        test_tma_load[
            IntTuple(64, 128),
            Layout.row_major(48, 128),
            IntTuple(48, 128),
            DType.bfloat16,
            OOB_access=True,
        ](ctx)

        test_tma_load[
            IntTuple(3, 16, 64),
            Layout.row_major(2, 16, 64),
            IntTuple(2, 16, 64),
            DType.bfloat16,
            OOB_access=True,
        ](ctx)
