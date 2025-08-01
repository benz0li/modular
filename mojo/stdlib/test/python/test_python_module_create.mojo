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

from python import Python


def test_create_module():
    var module = Python.create_module("test_module")

    # TODO: inspect properties about the module
    # First though, let's see if we can even import it
    # var imported_module = Python.import_module(module_name)
    #
    # _ = module_name


def main():
    test_create_module()
