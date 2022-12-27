#pragma once

#include "MemorySSA.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/IR/Operation.h"

#include <mlir/Pass/AnalysisManager.h>
#include <optional>

namespace rust_compiler::analysis {

class MemorySSAAnalysis {
public:
  MemorySSAAnalysis(mlir::Operation *op, ::mlir::AnalysisManager &am);
  MemorySSAAnalysis(const MemorySSAAnalysis &) = delete;

  ::mlir::LogicalResult optimizeUses();

  static bool
  isInvalidated(const ::mlir::AnalysisManager::PreservedAnalyses &pa);

  std::optional<MemorySSA> memssa;
  mlir::AliasAnalysis *aliasAnalysis = nullptr;
};

} // namespace rust_compiler::analysis
