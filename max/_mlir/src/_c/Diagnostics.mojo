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
#
# GENERATED FILE, DO NOT EDIT!
#
# Last generated by joe at 2024-09-19 16:25:28.063109 with command
# ```
#   ./utils/mojo/bindings-scripts/mlir/generate_mlir_mojo_bindings.sh
# ```
#
# ===----------------------------------------------------------------------=== #


from io.write import _WriteBufferStack

from .ffi import MLIR_func
from .IR import *
from .Support import *

# ===-- mlir-c/Diagnostics.h - MLIR Diagnostic subsystem C API ----*- C -*-===//
#
#  Part of the LLVM Project, under the Apache License v2.0 with LLVM
#  Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===//
#
#  This header declares the C APIs accessing MLIR Diagnostics subsystem.
#
# ===----------------------------------------------------------------------===//


@register_passable("trivial")
struct MlirDiagnostic:
    """An opaque reference to a diagnostic, always owned by the diagnostics engine
    (context). Must not be stored outside of the diagnostic handler."""

    var ptr: OpaquePointer


@fieldwise_init
@register_passable("trivial")
struct MlirDiagnosticSeverity(Copyable, Movable):
    """Severity of a diagnostic."""

    var value: Int8


alias MlirDiagnosticError = MlirDiagnosticSeverity(0)
alias MlirDiagnosticWarning = MlirDiagnosticSeverity(1)
alias MlirDiagnosticNote = MlirDiagnosticSeverity(2)
alias MlirDiagnosticRemark = MlirDiagnosticSeverity(3)

# Diagnostic handler type. Accepts a reference to a diagnostic, which is only
# guaranteed to be live during the call. The handler is passed the `userData`
# that was provided when the handler was attached to a context. If the handler
# processed the diagnostic completely, it is expected to return success.
# Otherwise, it is expected to return failure to indicate that other handlers
# should attempt to process the diagnostic.
alias MlirDiagnosticHandler = fn (
    MlirDiagnostic, OpaquePointer
) -> MlirLogicalResult


fn mlirDiagnosticPrint[W: Writer](mut writer: W, diagnostic: MlirDiagnostic):
    """Prints a diagnostic using the provided callback."""
    var buffer = _WriteBufferStack(writer)
    MLIR_func["mlirDiagnosticPrint", NoneType._mlir_type](
        diagnostic, write_buffered_callback[W], UnsafePointer(to=buffer)
    )
    buffer.flush()


fn mlirDiagnosticGetLocation(diagnostic: MlirDiagnostic) -> MlirLocation:
    """Returns the location at which the diagnostic is reported."""
    return MLIR_func["mlirDiagnosticGetLocation", MlirLocation](diagnostic)


fn mlirDiagnosticGetSeverity(
    diagnostic: MlirDiagnostic,
) -> MlirDiagnosticSeverity:
    """Returns the severity of the diagnostic."""
    return MLIR_func["mlirDiagnosticGetSeverity", MlirDiagnosticSeverity](
        diagnostic
    )


fn mlirDiagnosticGetNumNotes(diagnostic: MlirDiagnostic) -> Int:
    """Returns the number of notes attached to the diagnostic."""
    return MLIR_func["mlirDiagnosticGetNumNotes", Int](diagnostic)


fn mlirDiagnosticGetNote(
    diagnostic: MlirDiagnostic, pos: Int
) -> MlirDiagnostic:
    """Returns `pos`-th note attached to the diagnostic. Expects `pos` to be a
    valid zero-based index into the list of notes."""
    return MLIR_func["mlirDiagnosticGetNote", MlirDiagnostic](diagnostic, pos)


#  Attaches the diagnostic handler to the context. Handlers are invoked in the
#  reverse order of attachment until one of them processes the diagnostic
#  completely. When a handler is invoked it is passed the `userData` that was
#  provided when it was attached. If non-NULL, `deleteUserData` is called once
#  the system no longer needs to call the handler (for instance after the
#  handler is detached or the context is destroyed). Returns an identifier that
#  can be used to detach the handler.


fn mlirContextAttachDiagnosticHandler(
    context: MlirContext,
    handler: MlirDiagnosticHandler,
    user_data: OpaquePointer,
    delete_user_data: fn (OpaquePointer) -> None,
) -> MlirDiagnosticHandlerID:
    """Attaches the diagnostic handler to the context. Handlers are invoked in the
    reverse order of attachment until one of them processes the diagnostic
    completely. When a handler is invoked it is passed the `userData` that was
    provided when it was attached. If non-NULL, `deleteUserData` is called once
    the system no longer needs to call the handler (for instance after the
    handler is detached or the context is destroyed). Returns an identifier that
    can be used to detach the handler."""
    return MLIR_func[
        "mlirContextAttachDiagnosticHandler", MlirDiagnosticHandlerID
    ](context, handler, user_data, delete_user_data)


fn mlirContextDetachDiagnosticHandler(
    context: MlirContext, id: MlirDiagnosticHandlerID
) -> None:
    """Detaches an attached diagnostic handler from the context given its
    identifier."""
    return MLIR_func["mlirContextDetachDiagnosticHandler", NoneType._mlir_type](
        context, id
    )


fn mlirEmitError(location: MlirLocation, message: UnsafePointer[Int8]) -> None:
    """Emits an error at the given location through the diagnostics engine. Used
    for testing purposes."""
    return MLIR_func["mlirEmitError", NoneType._mlir_type](location, message)


# ===----------------------------------------------------------------------=== #
#     Codegen: Remaining symbols
# ===----------------------------------------------------------------------=== #

# Opaque identifier of a diagnostic handler, useful to detach a handler.
alias MlirDiagnosticHandlerID = UInt64
