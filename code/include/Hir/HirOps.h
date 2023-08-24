#pragma once

#include "Hir/HirEnum.h"
#include "Hir/HirOpsInterfaces.h"
#include "Hir/HirString.h"
#include "Hir/HirTypes.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Interfaces/LoopLikeInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Rewrite/FrozenRewritePatternSet.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <type_traits>

/// Include the auto-generated header file containing the declarations of the
/// Hir operations.
#define GET_OP_CLASSES
#include "Hir/HirOps.h.inc"
