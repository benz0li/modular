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

from builtin.device_passable import DevicePassable


trait CopyPolicy(DevicePassable, ImplicitlyCopyable):
    """
    The CopyPolicy trait defines requirements needed for a tensor to be copied.

    These requirements check the compatibility of the source and destination tensors.
    """

    @staticmethod
    fn verify_source_tensor(src: LayoutTensor):
        """
        A static function that verifies the source tensor
        is compatible with the copy operation. If the tensor is not valid
        compilation will fail.

        Args:
            src: The source tensor that will be copied from.
        """
        ...

    @staticmethod
    fn verify_destination_tensor(dst: LayoutTensor):
        """
        A static function that verifies the destination tensor
        is compatible with the copy operation. If the tensor is not valid
        compilation will fail.

        Args:
            dst: The destination tensor that will be copied to.
        """
        ...
