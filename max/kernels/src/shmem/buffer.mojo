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

from sys import has_nvidia_gpu_accelerator, CompilationTarget
from sys.ffi import external_call
from gpu.host import DeviceContext
from gpu.host.device_context import _checked, _DeviceContextPtr

from .host import shmem_malloc, shmem_free


struct SHMEMBuffer[dtype: DType](Sized):
    var _data: UnsafePointer[Scalar[dtype]]
    var _ctx_ptr: _DeviceContextPtr
    var _size: Int

    @doc_private
    @always_inline
    fn __init__(
        out self,
        ctx: DeviceContext,
        size: Int,
    ) raises:
        @parameter
        if has_nvidia_gpu_accelerator():
            self._data = shmem_malloc[dtype](size)
            self._ctx_ptr = ctx._handle
            self._size = size
        else:
            CompilationTarget.unsupported_target_error[
                operation="SHMEMBuffer.__init__",
            ]()
            self._data = UnsafePointer[Scalar[dtype]]()
            self._ctx_ptr = ctx._handle
            self._size = size

    fn __del__(owned self):
        shmem_free(self._data)

    fn enqueue_copy_to(self, dst_ptr: UnsafePointer[Scalar[dtype]]) raises:
        """Enqueues an asynchronous copy from this buffer to host memory.

        This method schedules a memory copy operation from this device buffer to the
        specified host memory location. The operation is asynchronous and will be
        executed in the stream associated with this buffer's context.

        Args:
            dst_ptr: Pointer to the destination host memory location.
        """
        _checked(
            external_call[
                "AsyncRT_DeviceContext_DtoH_async_sized",
                UnsafePointer[Byte],
                _DeviceContextPtr,
                UnsafePointer[Scalar[dtype]],
                UnsafePointer[Scalar[dtype]],
                Int,
            ](
                self._ctx_ptr,
                dst_ptr,
                self._data,
                self._size,
            )
        )

    fn __len__(self) -> Int:
        return self._size

    fn unsafe_ptr(self) -> UnsafePointer[Scalar[dtype]]:
        return self._data
