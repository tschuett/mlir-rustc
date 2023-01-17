#pragma once

#include <mlir/IR/FunctionInterfaces.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <type_traits>

#include "Mir/MirAttr.h"
#include "Mir/MirTypes.h"
#include "Mir/MirInterfaces.h"

namespace rust_compiler::Mir {}

/// Include the auto-generated header file containing the declarations of the
/// Mir operations.
#define GET_OP_CLASSES
#include "Mir/MirOps.h.inc"
