#pragma once

#include "Analysis/MemorySSA/MemorySSANodes.h"
#include "Analysis/MemorySSA/MemorySSAWalker.h"

#include <mlir/Analysis/AliasAnalysis.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/AnalysisManager.h>
#include <vector>

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
  void analyzeFunction(mlir::func::FuncOp *funcOp);
  std::optional<mlir::AliasResult> mayAlias(mlir::Operation *a,
                                            mlir::Operation *b);
  bool hasMemoryEffects(mlir::Operation &op);
  bool hasMemoryWriteEffect(mlir::Operation &op);
  bool hasMemoryReadEffect(mlir::Operation &op);
  bool hasCallEffects(mlir::Operation &op);

  std::shared_ptr<Node> createDef(mlir::Operation *, std::shared_ptr<Node> arg);
  std::shared_ptr<Node> createUse(mlir::Operation *, std::shared_ptr<Node> arg);
  std::shared_ptr<Node> createPhi(mlir::Operation *,
                                  llvm::ArrayRef<std::shared_ptr<Node>> args);
  std::shared_ptr<Node> getRoot();
  std::shared_ptr<Node> getTerm();

  mlir::ModuleOp module;
  mlir::AliasAnalysis *aliasAnalysis = nullptr;

  MemorySSAWalker *Walker = nullptr;

  std::vector<std::shared_ptr<Node>> nodes;
  std::shared_ptr<Node> root = nullptr;
  std::shared_ptr<Node> term = nullptr;
};

} // namespace rust_compiler::analysis
