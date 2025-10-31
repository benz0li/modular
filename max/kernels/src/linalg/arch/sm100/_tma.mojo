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
"""Tensor Memory Accelerator (TMA) utilities for NVIDIA SM100 architecture.

This module provides TMA descriptor creation and management for efficient memory
transfers on NVIDIA Blackwell (SM100) GPUs using the Tensor Memory Accelerator.
"""

from utils.index import IndexList
from gpu.host._tensormap import TensorMap, SwizzleMode, create_tensormap
from gpu.memory import (
    AddressSpace,
    cp_async_bulk_tensor_shared_cluster_global,
)
from layout.int_tuple import IntTuple, depth, to_index_list, product
from layout.runtime_tuple import to_index_list as runtime_tuple_to_index_list
from layout.layout_tensor import LayoutTensor
from layout.layout import Layout, zipped_divide, blocked_product
from layout.copy import CopyPolicy
from layout.tma_async import SharedMemBarrier
from gpu.host import DeviceBuffer, DeviceContext
from sys import size_of
from memory.pointer import AddressSpace as _AddressSpace
from layout.swizzle import Swizzle
from bit import log2_floor


struct TMADescriptor[
    dtype: DType, tile_shape: IntTuple, swizzle_mode: SwizzleMode
](ImplicitlyCopyable, Movable):
    var tensormap: TensorMap

    @always_inline
    @implicit
    fn __init__(out self, tensormap: TensorMap):
        """
        Initializes a new TMADescriptor with the provided TMA tensormap.

        Args:
            tensormap: The TMA tensormap that defines the memory access pattern.
        """
        self.tensormap = tensormap

    @always_inline
    fn __copyinit__(out self, other: Self):
        """
        Copy initializes this `TMADescriptor` from another instance.

        Args:
            other: The other `TMADescriptor` instance to copy from.
        """
        self.tensormap = other.tensormap


fn create_tma_descriptor[
    dtype: DType,
    tile_shape: IntTuple,
    /,
    *,
    swizzle_mode: SwizzleMode = SwizzleMode.NONE,
](
    gmem_tensor: LayoutTensor[
        dtype, *_, address_space = AddressSpace.GENERIC, **_
    ],
    ctx: DeviceContext,
) raises -> TMADescriptor[dtype, tile_shape, swizzle_mode]:
    """
    Creates a new TMADescriptor with the specified tile shape.

    Parameters:
        dtype: Data type of the tensor elements.
        tile_shape: Shape of the tile for TMA operations.
        swizzle_mode: Swizzle mode for shared memory access pattern optimization.

    Args:
        gmem_tensor: The tensor in global memory to create the TMA descriptor for.
        ctx: The device context to create the TMA descriptor.

    Returns:
        A new TMADescriptor configured for the specified tile shape and buffer.

    Raises:
        If the TMA descriptor creation fails.
    """

    constrained[depth(tile_shape) == 1, "Tile shape must be a flat tuple"]()

    alias rank = len(tile_shape)
    constrained[
        rank == gmem_tensor.rank,
        "Tile shape and input tensor's rank must match",
    ]()

    # Convert IntTuple to IndexList for create_tensormap
    return TMADescriptor[dtype, tile_shape, swizzle_mode](
        create_tensormap(
            DeviceBuffer(
                ctx,
                gmem_tensor.ptr.mut_cast[True]().address_space_cast[
                    AddressSpace.GENERIC
                ](),
                1,
                owning=False,
            ),
            runtime_tuple_to_index_list[rank](gmem_tensor.runtime_layout.shape),
            runtime_tuple_to_index_list[rank](
                gmem_tensor.runtime_layout.stride
            ),
            to_index_list[rank](tile_shape),
            swizzle_mode,
        ),
    )


struct TMALoad[
    dtype: DType,
    tile_shape: IntTuple,
    swizzle_mode: SwizzleMode = SwizzleMode.NONE,
](CopyPolicy):
    var descriptor: TMADescriptor[dtype, tile_shape, swizzle_mode]

    alias smem_alignment = 128

    @always_inline
    @implicit
    fn __init__(
        out self, ref descriptor: TMADescriptor[dtype, tile_shape, swizzle_mode]
    ):
        self.descriptor = descriptor

    @staticmethod
    fn verify_destination_tensor(dst: LayoutTensor):
        constrained[
            dtype == dst.dtype,
            String("type mismatch: expected ", dtype, " passed in ", dst.dtype),
        ]()

        constrained[
            dst.address_space == AddressSpace.SHARED,
            String(
                "address space mismatch: expected ",
                AddressSpace.SHARED,
                " passed in ",
                dst.address_space,
            ),
        ]()

        constrained[
            dst.alignment % Self.smem_alignment == 0,
            String(
                "alignment mismatch: expected ",
                Self.smem_alignment,
                " passed in ",
                dst.alignment,
            ),
        ]()

    @staticmethod
    fn verify_source_tensor(src: LayoutTensor):
        constrained[
            src.address_space == AddressSpace.GLOBAL,
            String(
                "address space mismatch: expected ",
                AddressSpace.GLOBAL,
                " passed in ",
                src.address_space,
            ),
        ]()

    @staticmethod
    fn layout_is_tma_compatible[repeat_pattern: Layout]() -> Bool:
        alias shape = repeat_pattern.shape
        alias stride = repeat_pattern.stride

        @parameter
        for i in range(len(shape)):
            alias current_shape = product(shape[i])
            alias current_stride = product(stride[i]) * size_of[dtype]()

            @parameter
            if current_shape > 1 and current_stride % Self.smem_alignment != 0:
                return False

        return True

    @staticmethod
    fn get_2D_smem_layout[m: Int, n: Int]() -> Layout:
        alias desc_layout = Layout.row_major(tile_shape)

        alias blocked_smem_layout = blocked_product(
            desc_layout,
            Layout.row_major(m, n),
            coalesce_output=True,
        )

        # check if layout is TMA compatible
        _ = Self.get_repeat_pattern[blocked_smem_layout]()

        return blocked_smem_layout

    @staticmethod
    fn get_repeat_pattern[
        dst_layout: Layout, check_tma_compatibility: Bool = True
    ]() -> Layout:
        alias descriptor_layout = Layout(tile_shape)  # uses column-major layout
        alias repeat_pattern = zipped_divide(dst_layout, descriptor_layout)[1]

        constrained[
            Self.layout_is_tma_compatible[repeat_pattern]()
            or repeat_pattern.size() == 1
            or check_tma_compatibility == False,
            "Layout does not respect TMA bank alignment",
        ]()

        return repeat_pattern


alias UInt32Indices[rank: Int] = IndexList[rank, element_type = DType.uint32]
alias MBarPtr = UnsafePointer[
    SharedMemBarrier, address_space = AddressSpace.SHARED
]


fn copy[
    rank: Int, //,
    cta_group: Int = 1,
](
    policy: TMALoad[*_],
    dst: LayoutTensor[_, _, address_space = AddressSpace.SHARED, *_, **_],
    mbar_ptr: MBarPtr,
    coords: UInt32Indices[rank],
):
    # Loads data from global memory to shared memory using the TMA operation. Works for dimensions of any size.
    # `policy` contains the descriptor. `coords` tells us the starting coordinate in the global memory tensor.

    # The TMA Operation can load data seperated by strides but will always write that data
    # to shared memory in a contiguous block.

    # For example assume we have this matrix in global memory,

    # [[A B C D]
    #  [E F G H]
    #  [I J K L]
    #  [M N O P]]

    # a preallocated smem ptr the same size as the matrix,
    # and a descriptor with shape [2, 2].

    # Now assume that we do a tma load from gmem coordinate (0, 0)
    # to ptr + 0.

    # Our ptr will now look like this:

    # [A B E F _ _ _ _ _ _ _ _ _ _ _ _]

    # And a naive row major layout will make the smem matrix look like this:

    # [[A B E F]
    #  [_ _ _ _]
    #  [_ _ _ _]
    #  [_ _ _ _]]

    # with a naive layout the raw memory will appear to be stored incorrectly. So when
    # doing loads with a width less than the tile width, it's important to use a nested layout.

    alias dst_layout = dst.layout

    policy.verify_destination_tensor(dst)

    # The coalesced layout is the row major version of any nested / normal layout.
    # It will have the same shape as the nested layout but will be in row major order.
    alias coalesced_shape = dst_layout.shape.product_flatten()
    alias coalesced_layout = Layout.row_major(coalesced_shape)

    # The repeat pattern tells us hoy many times our policy tile (descriptor)
    # can be repeated over the provided layout, in this case our destination
    # layout and our coalesced layout.
    alias dst_repeat_pattern = policy.get_repeat_pattern[dst_layout]()
    alias src_repeat_pattern = policy.get_repeat_pattern[
        coalesced_layout, check_tma_compatibility=False
    ]()

    constrained[
        dst_repeat_pattern.size() == src_repeat_pattern.size(),
        "Repeat patterns must have the same size",
    ]()

    alias num_copies = src_repeat_pattern.size()

    """
    Here's why we have 2 sets of offsets, repeat_patterns, and layouts.

    Assume we have this row major matrix in global memory.

    [[A B C D E F]
        [G H I J K L]
        [M N O P Q R]
        [S T U V W X]]

    Now assume we have this shared memory tensor with this 
    layout:

    ((2, 3), (2, 2)) : ((12, 4), (2, 1))

    [[0  1  4  5  8  9 ]
        [2  3  6  7  10 11]
        [12 13 16 17 20 21]
        [14 15 18 19 23 23]]

    and this descriptor: [2, 2]

    Our repeat pattern using this descriptor will consist of 6 blocks of 4 elements each.

    Each block will look like this:

    block 0: [[A B] [G H]] and smem offsets [[0 1] [2 3]]
    block 1: [[M N] [S T]] and smem offsets [[12 13] [14 15]]
    block 2: [[C D] [I J]] and smem offsets [[4 5] [6 7]]
    ...

    Now lets say we want to load block 3 into smem, for smem we would 
    load from offset 4, but if we start from offset 4 in global memory we would 
    load [E F K L] instead of [C D I J]. So we need a second layout that provides
    global memory offsets. To get this we just create a layout the same shape as 
    the shared memory layout but in row major order.
    """

    @parameter
    for i in range(num_copies):
        # The index i represents the tile we want to operate on.
        # We plug i into the repeat pattern to get the starting offset
        # of the desired tile.

        alias dst_copy_offset = dst_repeat_pattern(i)
        alias src_copy_offset = src_repeat_pattern(i)

        # The coordinate of a copy tile's starting point in dst
        # If repeat_pattern is (2, 2):(32, 4), the 2nd tile is at (1, 0) in (2, 2)
        # its offset's coordinates is (32, 0).
        alias offset_coords_tuple = coalesced_layout.idx2crd(src_copy_offset)
        alias offset_coords = to_index_list[rank, element_type = DType.uint32](
            offset_coords_tuple
        )

        # expects X, Y, Z coordinates
        var copy_tile_coords = (coords + offset_coords).reverse()

        cp_async_bulk_tensor_shared_cluster_global[cta_group=cta_group](
            dst.ptr + dst_copy_offset,
            UnsafePointer(to=policy.descriptor).bitcast[NoneType](),
            mbar_ptr,
            copy_tile_coords.cast[DType.int64](),
        )


@parameter
fn to_swizzle[dtype: DType, mode: SwizzleMode]() -> Swizzle:
    """Create swizzle based on predefined swizzle modes.

    Returns a swizzle pattern based on standard modes (32B, 64B,
    128B, none), adjusted for data type.

    Parameters:
        dtype: The data type of the elements.
        mode: The swizzle mode to use (TensorMapSwizzle enum).

    Returns:
        A `Swizzle` object configured by the specified mode.
    """
    alias type_size = size_of[dtype]()

    @parameter
    if mode in (
        SwizzleMode._128B,
        SwizzleMode._64B,
        SwizzleMode._32B,
        SwizzleMode.NONE,
    ):
        return Swizzle(Int(mode), log2_floor(16 // type_size), 3)
    else:
        constrained[False, "Only support 32B, 64B, 128B, or no swizzle"]()
        return Swizzle(0, 0, 0)
