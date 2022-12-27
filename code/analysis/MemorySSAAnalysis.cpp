#include "Analysis/MemorySSA/MemorySSAAnalysis.h"

#include <mlir/Analysis/AliasAnalysis.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Operation.h>

namespace {
struct Meminfo {
  mlir::Value memref;
  mlir::ValueRange indices;
};
} // namespace

auto hasMemEffect(mlir::Operation &op) {
  struct Result {
    bool read = false;
    bool write = false;
  };

  Result ret;
  if (auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op)) {
    if (effects.hasEffect<mlir::MemoryEffects::Write>())
      ret.write = true;

    if (effects.hasEffect<mlir::MemoryEffects::Read>())
      ret.read = true;
  } else if (op.hasTrait<mlir::OpTrait::HasRecursiveSideEffects>()) {
    ret.write = true;
  }
  return ret;
}

static std::optional<Meminfo> getMeminfo(mlir::Operation *op) {
  assert(nullptr != op);
  if (auto load = mlir::dyn_cast<mlir::memref::LoadOp>(op))
    return Meminfo{load.memref(), load.indices()};

  if (auto store = mlir::dyn_cast<mlir::memref::StoreOp>(op))
    return Meminfo{store.memref(), store.indices()};

  return std::nullopt;
}

rust_compiler::analysis::MemorySSAAnalysis::MemorySSAAnalysis(
    mlir::Operation *op, ::mlir::AnalysisManager &am) {
  if (op->getNumRegions() != 1)
    return;

  memssa = buildMemorySSA(op->getRegion(0));
  if (memssa) {
    aliasAnalysis = &am.getAnalysis<::mlir::AliasAnalysis>();
    (void)optimizeUses();
  }
}

::mlir::LogicalResult
rust_compiler::analysis::MemorySSAAnalysis::optimizeUses() {
  if (memssa) {
    assert(nullptr != aliasAnalysis);
    auto mayAlias = [&](mlir::Operation *op1, mlir::Operation *op2) {
      auto info1 = getMeminfo(op1);
      if (!info1)
        return true;

      auto info2 = getMeminfo(op2);
      if (!info2)
        return true;

      auto memref1 = info1->memref;
      auto memref2 = info2->memref;
      assert(memref1);
      assert(memref2);
      auto result = aliasAnalysis->alias(memref1, memref2);
      return !result.isNo();
    };
    return memssa->optimizeUses(mayAlias);
  }
  return ::mlir::failure();
}

bool rust_compiler::analysis::MemorySSAAnalysis::isInvalidated(
    const ::mlir::AnalysisManager::PreservedAnalyses &pa) {
  return !pa.isPreserved<MemorySSAAnalysis>() ||
         !pa.isPreserved<mlir::AliasAnalysis>();
}
