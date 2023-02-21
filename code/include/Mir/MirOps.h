#pragma once

#include "Mir/MirAttr.h"
#include "Mir/MirDialect.h"
#include "Mir/MirInterfaces.h"
#include "Mir/MirTypes.h"

#include <mlir/IR/FunctionInterfaces.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
//#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <type_traits>

namespace rust_compiler::Mir {}

/// Include the auto-generated header file containing the declarations of the
/// Mir operations.
#define GET_OP_CLASSES
#include "Mir/MirOps.h.inc"
