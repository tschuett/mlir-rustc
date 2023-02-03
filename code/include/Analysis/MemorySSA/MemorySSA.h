#pragma once

#include "MemorySSAWalker.h"

#include <mlir/Analysis/AliasAnalysis.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/AnalysisManager.h>
#include <vector>
#include <mlir/Dialect/Func/IR/FuncOps.h>

namespace rust_compiler::analysis {

class MemorySSA {
public:
  MemorySSA(mlir::ModuleOp _module, mlir::AnalysisManager &am) {
    aliasAnalysis = &am.getAnalysis<mlir::AliasAnalysis>();
    module = _module;
  }

  MemorySSA &operator=(const MemorySSA &) = delete;
  MemorySSA &operator=(MemorySSA &&) = default;

  void dump() const;
  void print(llvm::raw_ostream &) const;

  MemorySSAWalker *buildMemorySSA();

private:
  bool isFunction(mlir::Operation &op);
  void analyzeFunction(mlir::func::FuncOp *funcOp);

  std::optional<mlir::AliasResult> mayAlias(mlir::Operation *a,
                                            mlir::Operation *b);
  std::vector<mlir::Operation *> functionOps;

  mlir::ModuleOp module;
  mlir::AliasAnalysis *aliasAnalysis = nullptr;

  MemorySSAWalker *Walker = nullptr;
};

} // namespace rust_compiler::analysis
