#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

namespace rust_compiler::analysis {

class PointsTo {
  bool hasMemoryEffects(mlir::Operation *op);

  void analyzeFunction(mlir::func::FuncOp *op);

public:
  /// Points-to Analysis in Almost Linear Time by Bjarne Steensgaard
  void computePointsTo(mlir::ModuleOp &module);
};

} // namespace rust_compiler::analysis
