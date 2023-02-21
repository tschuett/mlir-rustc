#pragma once

#include "Hir/HirOpsInterfaces.h"

#include "Hir/HirStruct.h"
#include "Hir/HirEnum.h"
#include "Hir/HirString.h"

#include <mlir/Interfaces/InferTypeOpInterface.h>

#include <mlir/Dialect/Func/IR/FuncOps.h>
//#include <mlir/IR/FunctionInterfaces.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/ValueRange.h>
//#include <mlir/Interfaces/CallInterfaces.h>
//#include <mlir/Interfaces/ControlFlowInterfaces.h>
//#include <mlir/Interfaces/LoopLikeInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Rewrite/FrozenRewritePatternSet.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <type_traits>

/// Include the auto-generated header file containing the declarations of the
/// Hir operations.
#define GET_OP_CLASSES
#include "Hir/HirOps.h.inc"
