#pragma once

#include "Analysis/MemorySSA/MemorySSANodes.h"
#include "Analysis/MemorySSA/MemorySSAWalker.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Operation.h"

#include <mlir/Analysis/AliasAnalysis.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/AnalysisManager.h>
#include <vector>

namespace rust_compiler::analysis {

class MemorySSA {
  using AccessList = llvm::SmallVector<MemoryAccess *, 8>;
  using DefsList = llvm::SmallVector<MemoryDef *, 8>;

public:
  MemorySSA(mlir::func::FuncOp _fun, mlir::AnalysisManager &am) {
    aliasAnalysis = &am.getAnalysis<mlir::AliasAnalysis>();
    fun = _fun;
  }

  MemorySSA &operator=(const MemorySSA &) = delete;
  MemorySSA &operator=(MemorySSA &&) = default;

  void dump() const;
  void print(llvm::raw_ostream &) const;

  MemorySSAWalker *buildMemorySSA();

  MemoryUseOrDef *getMemoryAccess(const mlir::Operation *) const;
  MemoryPhi *getMemoryAccess(const mlir::Block *) const;

  /// Given two memory accesses in potentially different blocks,
  /// determine whether MemoryAccess \p A dominates MemoryAccess \p B.
  bool dominates(const MemoryAccess *A, const MemoryAccess *B) const;

  /// Return the list of MemoryAccess's for a given basic block.
  ///
  /// This list is not modifiable by the user.
  const AccessList *getBlockAccesses(const mlir::Block *BB) const;

  /// Return the list of MemoryDef's and MemoryPhi's for a given basic
  /// block.
  ///
  /// This list is not modifiable by the user.
  const DefsList *getBlockDefs(const mlir::Block *BB) const;

private:
  void analyzeFunction(mlir::func::FuncOp *funcOp);
  std::optional<mlir::AliasResult> mayAlias(mlir::Operation *a,
                                            mlir::Operation *b);
  bool hasMemoryEffects(mlir::Operation &op);
  bool hasMemoryWriteEffect(mlir::Operation &op);
  bool hasMemoryReadEffect(mlir::Operation &op);
  bool hasCallEffects(mlir::Operation &op);

  std::shared_ptr<MemoryDef> createDef(mlir::Operation *);
  std::shared_ptr<MemoryUse> createUse(mlir::Operation *);
  std::shared_ptr<MemoryPhi>
  createPhi(mlir::Operation *, llvm::ArrayRef<std::shared_ptr<Node>> args);
  std::shared_ptr<Node> getRoot();
  std::shared_ptr<Node> getTerm();

  mlir::func::FuncOp fun;
  mlir::AliasAnalysis *aliasAnalysis = nullptr;

  MemorySSAWalker *Walker = nullptr;

  std::vector<std::shared_ptr<Node>> nodes;
  std::shared_ptr<Node> root = nullptr;
  std::shared_ptr<Node> term = nullptr;

  std::unique_ptr<MemoryAccess> LiveOnEntryDef;

  // Memory SSA mappings
  llvm::DenseMap<const mlir::Value *, MemoryAccess *> ValueToMemoryAccess;

  llvm::DenseMap<mlir::Block *, std::unique_ptr<AccessList>> PerBlockAccesses;
  llvm::DenseMap<mlir::Block *, std::unique_ptr<DefsList>> PerBlockDefs;
};

} // namespace rust_compiler::analysis
