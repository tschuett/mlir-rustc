#pragma once

#include <mlir/IR/FunctionInterfaces.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <type_traits>
#include <mlir/Interfaces/LoopLikeInterface.h>

#include "Hir/HirInterfaces.h"

/// Include the auto-generated header file containing the declarations of the
/// Hir operations.
#define GET_OP_CLASSES
#include "Hir/HirOps.h.inc"
